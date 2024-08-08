import os
import os.path as osp
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset
from .gqa_eval_utils import eval_gqa


class VQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='gqa')

    def __init__(
        self,
        data_file,
        answer_file,
        image_folder,
        prompt_template,
        image_processor,
        tokenizer,
        prediction_file=None,
        pad_image_to_square=True,
        use_system=False,
        for_llava_prompt=False,
        metainfo=None,
        proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset),
    ):
        super().__init__(metainfo)
        self.data_file = data_file
        # Save detailed information for easy viewing
        self.answer_file = answer_file
        # solely for evaluation purposes
        self.prediction_file = prediction_file

        self.image_folder = image_folder
        self.use_system = use_system
        self.for_llava_prompt = for_llava_prompt
        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.data = self.load_data_list()

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def load_data_list(self):
        question_data = [json.loads(q) for q in open(os.path.expanduser(self.data_file), "r")]
        data_list = []
        for idx in range(len(question_data)):
            sample = question_data[idx]
            index = sample['question_id']
            image_path = sample['image']
            question = sample['text']

            data = {
                'img_id': idx,
                'index': index,
                'image_path': image_path,
                'question': question,
            }
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        answers_file = osp.join(work_dir, self.answer_file)
        ans_file = open(answers_file, "w")

        for pred_dict in results:
            idx = pred_dict["img_id"]
            gt_data = self.data[idx]

            ans_file.write(
                json.dumps(
                    {
                        "question_id": gt_data['index'],
                        "prompt": gt_data['question'],
                        "text": pred_dict['prediction'],
                        "metadata": {},
                    }
                )
                + "\n"
            )
        ans_file.close()

        with open(answers_file, 'r') as f:
            j = [json.loads(line) for line in f]
            # j = json.load(f)
            j.sort(key=lambda x: x['question_id'])
        with open(answers_file, 'w') as f:
            for data in j:
                json_data = json.dumps(data)
                f.write(json_data + '\n')

        if self.prediction_file is not None:
            cur_result = {}
            for line in open(answers_file):
                data = json.loads(line)
                qid = data['question_id']
                cur_result[f'v1_{qid}'] = data['text']

            with open(osp.join(work_dir, self.prediction_file), 'w') as f:
                json.dump(cur_result, f, indent=2)
        return {}
