import os
import os.path as osp
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset
from .gqa_eval_utils import eval_gqa


class MathDataset(BaseEvalDataset):
    METAINFO: dict = dict(name='gqa')

    def __init__(
        self,
        data_file,
        answer_file,
        image_folder,
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
        self.data_file = data_file
        # Save detailed information for easy viewing
        self.answer_file = answer_file

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
        question_data = json.load(open(os.path.expanduser(self.data_file), "r"))
        question_data = [dict(pid=pid, info=qs) for pid, qs in question_data.items()]

        data_list = []
        for idx in range(len(question_data)):
            sample = question_data[idx]
            index = sample['pid']
            image_path = sample['info']['image']
            question = self.create_one_query(
                problem=sample['info'],
                shot_num=0,
                shot_type='solution',
                use_caption=False,
            )
            data = {
                'img_id': idx,
                'index': index,
                'image_path': image_path,
                'question': question,
                'info': sample['info'],
            }
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    def create_one_query(self, problem, shot_num, shot_type, use_caption):
        ### [1] Demo prompt
        demo_prompt = ""

        ### [2] Test query
        # problem info
        question = problem['question']
        unit = problem['unit']
        choices = problem['choices']
        # caption = problem['caption']
        precision = problem['precision']
        question_type = problem['question_type']
        answer_type = problem['answer_type']

        # hint
        if shot_type == 'solution':
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        else:
            assert shot_type == 'code'
            hint_text = "Hint: Please generate a python code to solve the problem"

        # question
        question_text = f"Question: {question}"
        if unit:
            question_text += f" (Unit: {unit})"

        # choices
        if choices:
            # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
            texts = ["Choices:"]
            for i, choice in enumerate(choices):
                texts.append(f"({chr(ord('A')+i)}) {choice}")
            choices_text = "\n".join(texts)
        else:
            choices_text = ""

        # prompt
        if shot_type == 'solution':
            prompt = "Solution: "
        else:
            assert shot_type == 'code'
            prompt = "Python code: "

        elements = [hint_text, question_text, choices_text]
        test_query = "\n".join([e for e in elements if e != ""])

        ### [3] Final query
        query = demo_prompt + "\n\n" + test_query
        query = query.strip()
        return query

    @master_only
    def evaluate(self, results, work_dir):
        answers_file = osp.join(work_dir, self.answer_file)
        ans_file = open(answers_file, "w")

        for pred_dict in results:
            idx = pred_dict["img_id"]
            gt_data = self.data[idx]
            info = gt_data['info']
            info["query"] = gt_data['question'],
            info["response"] = pred_dict['prediction'],
            ans_file.write(json.dumps(info) + "\n")
        ans_file.close()

        return {}
