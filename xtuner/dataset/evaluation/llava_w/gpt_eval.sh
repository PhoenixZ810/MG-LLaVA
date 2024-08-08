set -e

PREDICTION=$1

DIR=$(dirname "$PREDICTION")
OUTPUTPATH="$DIR/llava_w_gpt_eval.jsonl"

srun -p llmit --time 1-00:00 \
    python xtuner/dataset/evaluation/llava_w/eval_gpt_review_bench.py \
    --question llava-bench-in-the-wild/questions.jsonl \
    --context llava-bench-in-the-wild/context.jsonl \
    --rule xtuner/dataset/evaluation/llava_w/rule.json \
    --answer-list \
        llava-bench-in-the-wild/answers_gpt4.jsonl \
        $PREDICTION \
    --output $OUTPUTPATH

srun -p llmit --time 1-00:00 \
    python xtuner/dataset/evaluation/llava_w/summarize_gpt_review.py -f $OUTPUTPATH