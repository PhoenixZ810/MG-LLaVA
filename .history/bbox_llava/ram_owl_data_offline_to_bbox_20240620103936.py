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


def get_rank_and_world_size():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return local_rank, world_size


# # 预训练数据
# image_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/llava_images'
# input_json_path = '/mnt/petrelfs/share_data/huanghaian/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
# save_json_path = './blip_laion_cc_sbu_558k_only_bbox.json'

# 指令微调数据
# image_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/LLaVA-Pretrain/images'
# input_json_path = '/mnt/petrelfs/share_data/huanghaian/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
# save_json_path = './llava_v1_5_mix665k_only_bbox.json'

# allava vflan dataset
# image_root = '/mnt/petrelfs/share_data/zhaoxiangyu'
# input_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/allava_vflan/ALLaVA-Instruct-VFLAN-4V-fixed.json'
# save_json_path = './allava-instruct-vflan4v_203k_only_box.json'

# image_root = 'data/mix'
# input_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/624bd4c5fedc2209cf952eedf75712413d8d912c/allava_vflan/ALLaVA-Caption-VFLAN-4V-fixed.json'
# save_json_path = './allava-caption-vflan4v_203k_only_box.json'

# allava laion dataset
# image_root = '/mnt/hwfile/mm_dev/zhaoxiangyu/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/624bd4c5fedc2209cf952eedf75712413d8d912c'
# input_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/ALLaVA-Instruct-LAION-4V-fixed.json'
# save_json_path = './allava-instruct-laion4v_489k_only_box.json'

# image_root = 'data/mix'
# input_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/624bd4c5fedc2209cf952eedf75712413d8d912c/allava_laion/ALLaVA-Caption-LAION-4V-From-MGM-fixed.json'
# save_json_path = './allava-caption-laion4v_505k_only_box.json'


# image_root = 'data/mix'
# input_json_path = 'mgm_ai2d_doc_dvqa_share_25969.json'
# save_json_path = 'mgm_ai2d_doc_dvqa_share_25969_only_box.json'

is_test=False

# test data
image_root = 'data/mix/vg'
input_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/tallyqa/test.json'
save_json_path = './tallyqa_38k_only_box.json'
is_test = True

is_debug=False
is_save_vis = False
save_path = './vis'

if is_save_vis:
    mkdir_or_exist(save_path)
    visualizer = Visualizer()

pth_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/recognize-anything-plus-model/ram_plus_swin_large_14m.pth'
image_size = 384
owl_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/models--google--owlv2-large-patch14-ensemble/snapshots/d638f16c163f70a8b6bd643b2ddbfc8be2c34807/'


if __name__ == '__main__':
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')

    mkdir_or_exist(osp.dirname(save_json_path))

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

    json_data = json.load(open(input_json_path))

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
        if not is_test:
            ids = data_sample['id']
        else:
            ids = data_sample['question_id']
        if isinstance(ids, int):
            ids = str(ids)
        curr_dict = {'id': ids}

        image_file = data_sample['image']
        image_file = os.path.join(image_root, image_file)

        if not os.path.exists(image_file):
            curr_dict['image'] = image_file
            curr_dict['boxes'] = []
            curr_dict['scores'] = []
            curr_dict['labels'] = []
            out_results.append(curr_dict)
            print(i, data_sample)
            continue
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
                visualizer.draw_texts(str(round(score, 3))+' | ' + label, positions=np.array(box[:2]).reshape(-1, 2), colors='r')

            drawn_img = visualizer.get_image()
            cv2.imwrite(os.path.join(save_path, f'{ids}.jpg'), drawn_img[...,::-1])

    if world_size > 1:
        dist.barrier()

    out_results = collect_results(out_results, len(json_data))

    if rank == 0:
        with open(save_json_path, 'w') as f:
            json.dump(out_results, f)