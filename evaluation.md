## Evaluation without GPT4
The evaluation metrics are specified in the sft configuration, including MMBench, SEED, SQA, AI2D, TextVQA, POPE, GQA, VQAv2,MMVet and additional ones. All evaluation config are exhibited in [Vicuna7B-sft config](mg_llava/config/vicuna/fuse_more_vicuna7b_clip_L_14_336_sft_padding.py) and you can set what you want to evaluate in `test_dataset`, as shown below:

  ```
  test_dataset = [
      dict(
          type=MultipleChoiceDataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_MMB-TEST_BBOX_JSON',
              image_size_aux=image_size_aux,
              limit_num=limit_num,
          ),
          data_file='PATH_TO_MMB-DEV_TSV',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,
          pad_image_to_square=pad_image_to_square,
      ),
      dict(
          type=TextVQADataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_TEXTVQA-VAL_BBOX_JSON',
              image_size_aux=image_size_aux,
              limit_num=limit_num,
          ),
          data_file='textvqa/llava_textvqa_val_v051_ocr.jsonl',
          ann_file='text_vqa/TextVQA_0.5.1_val.json',
          image_folder='text_vqa/train_images',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,
          pad_image_to_square=pad_image_to_square,
      ),
      dict(
          type=MMEDataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_MME_BBOX_JSON',
              image_size_aux=image_size_aux,
              limit_num=limit_num,
          ),
          data_file='PATH_TO_MME_TSV',
          image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,

          pad_image_to_square=pad_image_to_square,
      ),
      dict(
          type=POPEDataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_COCO-POPE_BBOX_JSON',
              image_size_aux=image_size_aux,
              limit_num=limit_num,
          ),
          data_file=[
              'POPE/coco_pope_adversarial.json',
              'POPE/coco_pope_popular.json',
              'POPE/coco_pope_random.json',
          ],
          coco_val_path='coco/val2014/',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,
          pad_image_to_square=pad_image_to_square,
      ),
      dict(
          type=GQADataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_GQA_BBOX_JSON',
              image_size_aux=image_size_aux,
              limit_num=limit_num,
          ),
          question_file='gqa/llava_gqa_testdev_balanced.jsonl',
          answer_file='llava_gqa_testdev_balanced_merge.jsonl', # file name of predicted answer
          prediction_file='testdev_balanced_predictions.json', # file name of formatted predicted answer
          test_question_file='gqa/testdev_balanced_questions.json',
          image_folder='gqa/images',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,
          pad_image_to_square=pad_image_to_square,
      ),
      dict(
          type=VQAv2Dataset,
          proxy_eval_dataset=dict(
              type=MGLLaVAProxyEvalDataset,
              box_json_path='PATH_TO_VQA_BBOX_JSON',
              image_size_aux=image_size_aux,
          ),
          question_file='vqa/llava_vqav2_mscoco_test-dev2015.jsonl',
          answer_file='llava_vqav2_testdev_balanced_merge.jsonl', # file name of predicted answer
          test_file='vqa/llava_vqav2_mscoco_test2015.jsonl',
          prediction_file='vqav2_testdev_balanced_predictions.json', # file name of formatted predicted answer
          image_folder='vqa/vqav2_test2015',
          prompt_template=PROMPT_TEMPLATE.vicuna,
          tokenizer=tokenizer,
          image_processor=image_processor,
          pad_image_to_square=pad_image_to_square,
      ),
      ......
  ]
  ```

Before evaluation, you should modify the [test config](script/test.sh). Then run the following command:
```shell
bash script/test.sh
```
The prediction file and results will be saved in the `workdirs` directory. For VQAv2 and MMVet benchmarks, you should upload the result file to the official website for evaluation.

## Evaluation with GPT4

For benchmarks using GPT4 for evaluation, such as LLaVA-Bench, MMVP and MathVista, you can refer to the [Vicuna7B-gpt4 config](mg_llava/config/vicuna/fuse_more_vicuna7b_clip_L_14_336_gpt4_padding.py):

  ```
      dict(
        type=VQADataset,
        proxy_eval_dataset=dict(
            type=MGLLaVAProxyEvalDataset,
            box_json_path='PATH_TO_LLAVA_BENCH_IN_THE_WILD_BBOX_JSON',
            image_size_aux=image_size_aux,
        ),
        data_file='llava-bench-in-the-wild/questions.jsonl',
        answer_file='llava_w_prediction.jsonl',
        image_folder='llava-bench-in-the-wild/images',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MathDataset,
        proxy_eval_dataset=dict(
            type=MGLLaVAProxyEvalDataset,
            box_json_path='PATH_TO_MATHVISTA_BBOX_JSON',
            image_size_aux=image_size_aux,
        ),
        data_file='MathVista/test_mini.json',
        answer_file='mathvista_prediction.jsonl',
        image_folder='MathVista/',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MMVPDataset,
        proxy_eval_dataset=dict(
            type=MGLLaVAProxyEvalDataset,
            box_json_path='PATH_TO_MMVP_BBOX_JSON',
            image_size_aux=image_size_aux,
        ),
        data_file='MMVP/Questions.csv',
        answer_file='mmvp_prediction.jsonl',
        image_folder='MMVP/',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
```

After inference and saving the prediction file, the prediction file and results will be saved in the `workdirs` directory, please use the `gpt_eval.sh` in the response directory([LLaVA-W](xtuner/dataset/evaluation/llava_w/gpt_eval.sh), [MMVP](xtuner/dataset/evaluation/MMVP/gpt_eval.sh), [MathVista](xtuner/dataset/evaluation/math_vista/gpt_eval.sh)) to evaluate the results.

For example, if you want to evaluate MMVP, you should first add your private OpenAI API key and Base URL in the `xtuner/dataset/evaluation/MMVP/gpt_grader.py` file,.and then you can run the following command to get the final results:

```
bash xtuner/dataset/evaluation/MMVP/gpt_eval.sh workdirs/xxxxxxx/mmvp_prediction.jsonl
```