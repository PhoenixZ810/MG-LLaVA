import os
import os.path as osp
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.dataset.utils import decode_base64_to_image
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset


class VIDEOQADataset(BaseEvalDataset):

    METAINFO: dict = dict(name='video')

    def __init__(
        self,
        gt_question_file,
        gt_answer_file,
        pred_file,
        video_folder,
        prompt_template,
        image_processor,
        tokenizer,
        pad_image_to_square=True,
        use_system=False,
        for_llava_prompt=False,
        metainfo=None,
        proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset),
    ):
        super().__init__(metainfo)
        self.gt_question_file = gt_question_file
        self.gt_answer_file = gt_answer_file
        self.pred_file = pred_file
        self.video_folder = video_folder
        self.use_system = use_system
        self.for_llava_prompt = for_llava_prompt
        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.data = self.load_data_list()
        self.data_dict = {item['id']: item for item in self.data}

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def load_data_list(self):
        gt_question_data = json.load(open(self.gt_question_file))
        gt_answer_data = json.load(open(self.gt_answer_file))
        data_list = []
        for idx in range(len(gt_question_data)):
            sample = gt_question_data[idx]
            answer = gt_answer_data[idx]['answer']
            data = {
                'id': sample['question_id'],
                'video': sample['video_name'],
                'question': sample['question'],
                'answer': answer,
            }
            data_list.append(data)
            idx += 1

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        answers_file = osp.join(work_dir, self.pred_file)
        ans_file = open(answers_file, "w")
        results_file = []
        for pred_dict in results:
            question_id = pred_dict['img_id']
            results_file.append({
                        "id": question_id,
                        "question": self.data_dict[question_id]['question'],
                        "answer": self.data_dict[question_id]['answer'],
                        "pred": pred_dict['prediction'],
                    }
                )
        json.dump(results_file, ans_file)
        ans_file.close()
