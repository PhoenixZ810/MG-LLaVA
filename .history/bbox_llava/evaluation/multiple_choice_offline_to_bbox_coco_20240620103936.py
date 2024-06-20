import torch.distributed as dist
from mmengine.dist import collect_results
import os
import torch
import os.path as osp
from mmengine.utils import mkdir_or_exist

from PIL import Image
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
    'MMBench_DEV_EN.tsv',
    'MMBench_TEST_EN.tsv',
    'SEEDBench_IMG.tsv',
    'ScienceQA_VAL.tsv',
    'ScienceQA_TEST.tsv',
    'MMMU_DEV_VAL.tsv',
    'AI2D_TEST.tsv',
    'HallusionBench.tsv',
    'MME.tsv',
    'MMStar.tsv'
]

mme_image_folder = '/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release'
mmstar_root_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/datasets--Lin-Chen--MMStar/snapshots/mmstar'
save_path = 'data/eval_box_coco'


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

COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors' 'teddy bear',
    'hair drier',
    'toothbrush',
]
def get_image(image, df):
    while len(image) < 16:
        image = df[df['index'] == int(image)]['image'].values
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

    processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model.cuda()
    owl_model.eval()

    for d in data_name:
        print('=========================================================================')
        print(f'Processing {d}')
        if d == 'MMStar.tsv':
            data_file = osp.join(mmstar_root_path, d)
        else:
            data_file = osp.join(root_path, d)
        df = pd.read_csv(data_file, sep='\t')
        if d == 'MME.tsv' or d == 'HallusionBench.tsv':
            # 删除没有图片的数据
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
            if d == 'MME.tsv':
                image_path = df.iloc[i]['image_path']
                orig_image = Image.open(os.path.join(mme_image_folder, image_path))
            else:
                image = df.iloc[i]['image']
                try:
                    orig_image = get_image(image, df)
                except Exception as e:
                    curr_dict['boxes'] = []
                    curr_dict['scores'] = []
                    curr_dict['labels'] = []
                    out_results.append(curr_dict)
                    print(f'get no image! id:{i}')
                    continue

            # model forward

            orig_image = orig_image.convert("RGB")
            texts = ['a photo of ' + cate for cate in COCO_CATEGORIES]

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
            labels = [COCO_CATEGORIES[label] for label in labels]

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
