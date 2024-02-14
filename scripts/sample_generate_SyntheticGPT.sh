export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=16

id=$1
task=$2
pos_mode=$3
neg_mode=$4
generator=$5
retrieval=$6
model_name_or_path=$7
if echo "$generator" | grep -q "mistral"; then
    generation_batch_size=48
    scoring_batch_size=48
else
    generation_batch_size=12
    scoring_batch_size=24
fi
mkdir -p logs/$id/$task
echo start $id $task generation
python -u do_generation.py \
    --index $id \
    --task $task \
    --generator $generator \
    --retrieval $retrieval \
    --model_name_or_path $model_name_or_path \
    --pos_mode $pos_mode \
    --neg_mode $neg_mode \
    --do_sample \
    --batch_size $generation_batch_size \
    --max_new_tokens 200 \
    --output_dir ${id}_generated_${task}_test.json > logs/$id/$task/generate.log 2>&1

echo start $id $task scoring
python -u do_scoring.py \
    --index $id \
    --task $task \
    --generator $generator \
    --retrieval $retrieval \
    --model_name_or_path $model_name_or_path \
    --pos_mode $pos_mode \
    --neg_mode $neg_mode \
    --do_sample \
    --batch_size $scoring_batch_size \
    --max_new_tokens 200 \
    --output_dir ${id}_scored_${task}_test.json > logs/$id/$task/score.log 2>&1
