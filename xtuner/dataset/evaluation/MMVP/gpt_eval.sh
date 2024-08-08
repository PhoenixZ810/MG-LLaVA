set -e
answer_file=$1

srun -p llmit --time 1-00:00 \
    python xtuner/dataset/evaluation/MMVP/gpt_grader.py \
    --answer_file $answer_file