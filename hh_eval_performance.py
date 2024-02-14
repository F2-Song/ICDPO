import os
import argparse
import json
import tqdm
import torch

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--res_flag', default="", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    global_sample_num = 0
    global_reward = 0

    for file_name in [
        "harmless_base.json",
        "helpful_base.json",
    ]:
        save_path = os.path.join("inference_res", "{}_{}".format(args.res_flag, file_name))
        if not os.path.exists(save_path):
            continue
        with open(save_path, 'r', encoding='utf-8') as f:
            infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines())]

        avg_reward = 0
        for line in infer_data:
            avg_reward += line['infer']['score']
        
        global_sample_num += len(infer_data)
        global_reward += avg_reward

        avg_reward = avg_reward / len(infer_data)            
        print("Eval on {}".format(file_name))
        print("Avg Reward: {}".format(avg_reward))
        
    print("")
    print("Global Eval")
    print("Avg Reward: {}".format(global_reward / global_sample_num))