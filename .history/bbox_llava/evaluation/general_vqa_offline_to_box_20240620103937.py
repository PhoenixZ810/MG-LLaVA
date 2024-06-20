import torch.distributed as dist
from mmengine.dist import collect_results
import os
import torch
import os.path as osp
from mmengine.utils import mkdir_or_exist

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import json
import math
from tqdm import tqdm
from mmengine.visualization import Visualizer
import numpy as np
import cv2
import pandas as pd
from xtuner.dataset.utils import decode_base64_to_image


def get_rank_and_world_size():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return local_rank, world_size


# 预训练数据
root_path = '/mnt/petrelfs/share_data/huanghaian/LMUData'
data_name = [
    # 'DocVQA_VAL.tsv',
    'OCRVQA_TEST.tsv'
]

save_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/LUMDdata_box/'


is_debug = False
is_save_vis = False
save_vis_path = 'debug/vis'

if is_save_vis:
    for d in data_name:
        mkdir_or_exist(save_vis_path + '/' + d[:-4])
    visualizer = Visualizer()

pth_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/recognize-anything-plus-model/ram_plus_swin_large_14m.pth'
image_size = 384
owl_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/models--google--owlv2-large-patch14-ensemble/snapshots/d638f16c163f70a8b6bd643b2ddbfc8be2c34807/'


def get_image(image, df):
    while len(image) < 16:
        image = df[df['index'] == int(image)]['image'].values
        assert len(image) == 1
        image = image[0]
    image = decode_base64_to_image(image)
    return image
def get_image_ocr(image, df):
    while len(image) < 16:
        image = df[df['index'] == image]['image'].values
        assert len(image) == 1
        image = image[0]
    image = decode_base64_to_image(image)
    return image

if __name__ == '__main__':
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')

    mkdir_or_exist(osp.dirname(save_path))

    # build model
    transform = get_transform(image_size=image_size)
    ram_model = ram_plus(pretrained=pth_path, image_size=image_size, vit='swin_l')
    ram_model.cuda()
    ram_model.eval()

    processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model.cuda()
    owl_model.eval()

    for d in data_name:
        print('=========================================================================')
        print(f'Processing {d}')

        data_file = osp.join(root_path, d)
        if 'OCR' in data_file:
            ocr = True
        df = pd.read_csv(data_file, sep='\t')
        skip_noimg = True
        if skip_noimg:
            df = df[~pd.isna(df['image'])]

        results = []
        n_samples = len(df)
        per_rank_samples = math.ceil(n_samples / world_size)

        per_rank_ids = range(per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1)))

        if is_debug:
            assert world_size == 1, 'Debug mode only supports single process'
            per_rank_ids = range(0, 50)

        out_results = []
        for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
            index = df.iloc[i]['index']
            if isinstance(index, np.int64):
                curr_dict = {'id': int(index)}
            else:
                curr_dict = {'id': index}

            image = df.iloc[i]['image']
            try:
                if ocr:
                    orig_image = get_image_ocr(image, df)
                else:
                    orig_image = get_image(image, df)
                # model forward
                with torch.no_grad():
                    image = transform(orig_image).unsqueeze(0).cuda()
                    res = inference(image, ram_model)
            except Exception as e:
                curr_dict['boxes'] = []
                curr_dict['scores'] = []
                curr_dict['labels'] = []
                out_results.append(curr_dict)
                print(f'get no image! id:{i}')
                # raise
                continue


            tags = res[0].replace(' |', ',')
            tags = tags.lower()
            tags = tags.strip()
            tags = tags.split(',')
            tags = [tag.strip() for tag in tags]
            prompt_tags = ['a photo of ' + tag for tag in tags]

            orig_image = orig_image.convert("RGB")
            texts = [prompt_tags]
            with torch.no_grad():
                inputs = processor(text=texts, images=orig_image, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = owl_model(**inputs)

            max_wh = max(orig_image.size)
            target_sizes = torch.Tensor([[max_wh, max_wh]])
            results = processor.post_process_object_detection(
                outputs=outputs, threshold=0.2, target_sizes=target_sizes
            )

            index1 = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[index1]
            boxes, scores, labels = (
                results[index1]["boxes"],
                results[index1]["scores"],
                results[index1]["labels"],
            )

            boxes = boxes.int().cpu().numpy().tolist()
            scores = scores.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            labels = [tags[label] for label in labels]

            curr_dict['boxes'] = boxes
            curr_dict['scores'] = scores
            curr_dict['labels'] = labels

            out_results.append(curr_dict)

            if is_save_vis:
                visualizer.set_image(np.array(orig_image))

                for box, score, label in zip(boxes, scores, labels):
                    visualizer.draw_bboxes(np.array(box).reshape(-1, 4), edge_colors='r')
                    visualizer.draw_texts(
                        str(round(score, 3)) + ' | ' + label,
                        positions=np.array(box[:2]).reshape(-1, 2),
                        colors='r',
                    )

                drawn_img = visualizer.get_image()
                cv2.imwrite(os.path.join(save_vis_path + '/' + d[:-4], f'{index}.jpg'), drawn_img[..., ::-1])

        if world_size > 1:
            dist.barrier()

        out_results = collect_results(out_results, len(df))

        if rank == 0:
            save_json_path = osp.join(save_path, f'{d[:-4]}_only_bbox.json')
            with open(save_json_path, 'w') as f:
                json.dump(out_results, f)
