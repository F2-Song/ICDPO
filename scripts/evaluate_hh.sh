export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

id=$1
task=$2 # hh, hh_llama_chat
res_flag=${id}_scored_${task}

echo start $res_flag
python -u hh_eval_reward.py \
    --res_flag $res_flag > logs/${id}/rewarding_${task}.log 2>&1

python -u hh_eval_performance.py \
    --res_flag $res_flag > logs/${id}/performance_${task}.log 2>&1
