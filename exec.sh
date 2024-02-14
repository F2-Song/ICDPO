script=hh # SyntheticGPT
id=1
task=hh # hh, hh_llama_chat, SyntheticGPT, SyntheticGPT_llama_chat
pos_mode=icl
neg_mode=base # icl
generator=llama # llama2, mistral
retrieval=random # random, sbert
model_name_or_path=YOUR_MODEL_PATH

sh scripts/sample_generate_${script}.sh $id $task $pos_mode $neg_mode $generator $retrieval $model_name_or_path
sh scripts/evaluate_${script}.sh $id $task