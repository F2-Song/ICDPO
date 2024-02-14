from utils.config import args
from load_data.data_manager import Data_Manager
from utils.scoring_utils import score
import numpy as np

path = f"inference_res/cache/"
if "hh" in args.task:
    path += f"{args.index}_generated_{args.task}.json"
else:
    assert "SyntheticGPT" in args.task
    path += f"{args.index}_generated_{args.task}_test.json"

data_manager = Data_Manager(path, args.task)
# postpreprocess
for sample in data_manager.test_set:
    for candidate in sample["candidates"]:
        candidate["t_raw"] = candidate["t"]
        candidate["t"] = data_manager.postpreprocess(
            candidate["t"],
            do_early_truncation=True,
        )

mode_roles = {
    "pos": (args.pos_mode, args.pos_model_direction),
    "neg": (args.neg_mode, args.neg_model_direction),
}
pred_scores = score(
    data_manager, 
    args.pos_mode,
    args.neg_mode,
    args.pos_model_direction,
    args.neg_model_direction,
)

if mode_roles["pos"][0] is None and mode_roles["neg"][0] is None:
    raise Exception("No mode is specified!")

for mode in ["pos", "neg"]:
    if mode_roles[mode][0] == None:
        continue
    for sample, scores, mean_scores in zip(data_manager.test_set, pred_scores[mode]):
        for candidate_index in range(len(sample["candidates"])):
            sample["candidates"][candidate_index][f"{mode}_score"] = scores[candidate_index]
    sample[f"{mode}_role"] = mode_roles[mode][0]
    sample[f"{mode}_model_direction"] = mode_roles[mode][1]

for sample in data_manager.test_set:
    # choose the candidate response with the highest score
    for candidate in sample["candidates"]:
        if mode_roles["neg"][0] == None:
            # score is the candidate response with the highest pos_score
            candidate["score"] = candidate[f"pos_score"]
        elif mode_roles["pos"][0] == None:
            # score is the candidate response with the lowest neg_score
            candidate["score"] = -candidate[f"neg_score"]
        else:
            # choose the candidate response with the highest pos_score - neg_score
            candidate["score"] = candidate[f"pos_score"] - candidate[f"neg_score"]
    # get the index of the candidate response with the highest score
    chosen_index = np.argmax([candidate["score"] for candidate in sample["candidates"]])
    sample["infer"] = {
        "t": sample["candidates"][chosen_index]["t"],
        "index": int(chosen_index),
    }
data_manager.save_test_set(f"inference_res/cache/{args.output_dir}")