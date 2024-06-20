import torch.distributed as dist
from mmengine.dist import collect_results
import os
import torch
import os.path as osp
from mmengine.utils import mkdir_or_exist

from PIL import Image
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge('torch')
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


# 预训练数据
video_root = '/mnt/petrelfs/share_data/duanhaodong/'
# input_json_path = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/origin_train_json/valley_llavaimage.json'
frame_num = 8

save_json_path = f'video_valley_702k_frame{frame_num}_only_bbox_4.json'
video_part = 'pretrain_videos_part_4.txt'

# 指令微调数据
video_root = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data'
input_json_path = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/train_json/videochatgpt_llavaimage_tune_modify_shuffle_v2.json'
save_json_path = f'video_chatgpt_100k_frame{frame_num}_only_bbox.json'

# video 数据
video_root = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/VideoQA_Eval/MSRVTT_Zero_Shot_QA/videos/all'
input_json_path = (
    '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/VideoQA_Eval/MSRVTT_Zero_Shot_QA/test_q.json'
)
save_json_path = f'MSRVTT_frame{frame_num}_only_bbox.json'

is_debug = False
is_save_vis = False
save_path = './debug/vis_video'

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
    ram_model = ram_plus(pretrained=pth_path, image_size=image_size, vit='swin_l')
    ram_model.cuda()
    ram_model.eval()

    processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=owl_path)
    owl_model.cuda()
    owl_model.eval()

    json_data = json.load(open(input_json_path))

    videos = []
    video_files = []
    images = []

    # split pretrain data
    # for data in json_data:
    #     if 'video' in data:
    #         videos.append(data)
    #         video_files.append(data['video'])
    #     elif 'image' in data:
    #         images.append(data)

    ## split sft data
    # for data in json_data:
    #     if 'video' in data and data['video']:
    #         video_files.append(data['video'])
    # # print(len(video_files))
    # video_files_set = []
    # seen = set()
    # for data in json_data:
    #     if 'video' in data and data['video'] and data['video'] not in seen:
    #         seen.add(data['video'])
    #         video_files_set.append(data['video'])
    # # print(len(video_files_set))
    # videos = video_files_set
    # # print(len(list(set(videos))))
    # print(len(video_files), len(video_files_set), rank, world_size)
    ## split pretrain video files
    # n = len(video_files_set) // 3
    # list_parts = [video_files_set[i : i + n] for i in range(0, len(video_files_set), n)]
    # if len(video_files_set) % 3 > 0:
    #     list_parts[-1].extend(video_files_set[-(len(video_files_set) % 3) :])
    # for i, part in enumerate(list_parts):
    #     with open(f'pretrain_videos_part_{i+1}.txt', 'w') as file:
    #         file.write('\n'.join(part))
    # raise

    ## read split video files part
    # with open(video_part, 'r') as file:
    #     videos = file.readlines()
    #     videos = [line.rstrip() for line in videos]

    ## MSVD & MSRVTT
    for data in json_data:
        video_files.append(data['video_name'])
    # print(len(video_files))
    video_files_set = []
    seen = set()
    for data in json_data:
        if data['video_name'] not in seen:
            seen.add(data['video_name'])
            video_files_set.append(data['video_name'])
    # print(len(video_files_set))
    videos = video_files_set
    # print(len(list(set(videos))))
    print(len(video_files), len(video_files_set), rank, world_size)
    # videos = [i + '.avi' for i in videos] # MSVD
    videos = [i + '.mp4' for i in videos]

    results = []
    n_samples = len(videos)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1)))
    print(per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1)))
    if is_debug:
        assert world_size == 1, 'Debug mode only supports single process'
        per_rank_ids = range(0, 50)

    out_results = []
    for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = videos[i]

        # ids = data_sample['id']
        # if isinstance(data_sample['id'], int):
        #     ids = str(data_sample['id'])
        # curr_dict = {'id': ids}

        # video_file = data_sample['video']
        video_file = data_sample
        video_path = os.path.join(video_root, video_file)
        if not os.path.exists(video_path):
            curr_dict = {'video': video_file}
            curr_dict['frames'] = []
            out_results.append(curr_dict)
            print(i, data_sample)
            continue
        curr_dict = {'video': video_file}

        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        # print(duration)
        frame_id_list = np.linspace(0, duration - 1, frame_num, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        images = [Image.fromarray(frame.numpy()) for frame in video_data]

        # model forward
        # orig_image = Image.open(image_file)
        curr_dict['frames'] = []
        for idx, orig_image in enumerate(images):
            frame_dict = {}
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
            results = processor.post_process_object_detection(
                outputs=outputs, threshold=0.2, target_sizes=target_sizes
            )

            index = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[index]
            boxes, scores, labels = (
                results[index]["boxes"],
                results[index]["scores"],
                results[index]["labels"],
            )

            boxes = boxes.int().cpu().numpy().tolist()
            scores = scores.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            labels = [tags[label] for label in labels]

            frame_dict['boxes'] = boxes
            frame_dict['scores'] = scores
            frame_dict['labels'] = labels
            curr_dict['frames'].append(frame_dict)

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
                path_parts = video_file.split('/')
                name = '_'.join(path_parts[1:])[:-4]
                cv2.imwrite(os.path.join(save_path, f'{name}_{idx}.jpg'), drawn_img[..., ::-1])

        out_results.append(curr_dict)

    if world_size > 1:
        dist.barrier()

    out_results = collect_results(out_results, len(videos))

    if rank == 0:
        with open(save_json_path, 'w') as f:
            json.dump(out_results, f)
