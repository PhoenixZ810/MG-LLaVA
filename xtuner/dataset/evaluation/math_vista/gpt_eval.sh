set -e

input_file=$1
score_file="math_vista_score.json"

DIR=$(dirname "$input_file")
score_file_path="$DIR/$score_file"



srun -p llmit --time 1-00:00 \
    python xtuner/dataset/evaluation/math_vista/extract_answer.py \
    --output_file $input_file \

srun -p llmit --time 1-00:00 \
    python xtuner/dataset/evaluation/math_vista/calculate_score.py \
    --output_file $input_file \
    --score_file $score_file_path \
    --gt_file MathVista/test_mini.json