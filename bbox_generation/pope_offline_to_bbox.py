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
from xtuner.dataset.evaluation.utils import load_jsonl


def get_rank_and_world_size():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return local_rank, world_size


# 预训练数据
data_file = [
    '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_adversarial.json',
    '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_popular.json',
    '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_random.json'
]
coco_val_path = '/mnt/petrelfs/share_data/linzhihao/dataset/coco/val2014/'
save_json_path = [
    'data/eval_box/pope/coco_pope_adversarial_only_bbox.json',
    'data/eval_box/pope/coco_pope_popular_only_bbox.json',
    'data/eval_box/pope/coco_pope_random_only_bbox.json',
]

is_debug = False
is_save_vis = False
save_vis_path = './vis'

if is_save_vis:
    for d in data_file:
        basename = osp.basename(d)
        mkdir_or_exist(save_vis_path + '/' + basename[:-5])
    visualizer = Visualizer()

pth_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/recognize-anything-plus-model/ram_plus_swin_large_14m.pth'
image_size = 384
owl_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/models--google--owlv2-large-patch14-ensemble/snapshots/d638f16c163f70a8b6bd643b2ddbfc8be2c34807/'

if __name__ == '__main__':
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')

    mkdir_or_exist(osp.dirname(save_json_path[0]))

    # build model
    transform = get_transform(image_size=image_size)
    ram_model = ram_plus(pretrained=pth_path,
                         image_size=image_size,
                         vit='swin_l')
    ram_model.cuda()
    ram_model.eval()

    processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model.cuda()
    owl_model.eval()

    for ind,d in enumerate(data_file):
        print('=========================================================================')
        print(f'Processing {d}')

        json_data = load_jsonl(d)

        results = []
        n_samples = len(json_data)
        per_rank_samples = math.ceil(n_samples / world_size)

        per_rank_ids = range(per_rank_samples * rank,
                             min(n_samples, per_rank_samples * (rank + 1)))

        if is_debug:
            assert world_size == 1, 'Debug mode only supports single process'
            per_rank_ids = range(0, 50)

        out_results = []
        for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
            data_sample = json_data[i]

            curr_dict = {'id': data_sample['question_id']}

            image_file = data_sample['image']
            image_file = os.path.join(coco_val_path, image_file)

            # model forward
            orig_image = Image.open(image_file)
            with torch.no_grad():
                image = transform(orig_image).unsqueeze(0).cuda()
                res = inference(image, ram_model)
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
            results = processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)

            index = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[index]
            boxes, scores, labels = results[index]["boxes"], results[index]["scores"], results[index]["labels"]

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
                    visualizer.draw_texts(str(round(score, 3)) + ' | ' + label, positions=np.array(box[:2]).reshape(-1, 2),
                                          colors='r')

                drawn_img = visualizer.get_image()
                cv2.imwrite(os.path.join(save_json_path[i], f'{i}.jpg'), drawn_img[..., ::-1])

        if world_size > 1:
            dist.barrier()

        out_results = collect_results(out_results, len(json_data))

        if rank == 0:
            with open(save_json_path[ind], 'w') as f:
                json.dump(out_results, f)
