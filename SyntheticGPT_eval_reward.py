import os
import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM
)
from peft import PeftConfig, PeftModel
import numpy as np
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--rank_sum', type=int, default=1)
    parser.add_argument('--res_flag', default="", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ['RANK'] = str(args.rank)
    rank = args.rank
    rank_sum = args.rank_sum
    import metrics
    setup_seed()
    
    torch.cuda.empty_cache()
    print(f"Rank {rank} is activated...")
    get_score, reward_batch_size = metrics.create_reward_fn()
    
    for file_name in [
        "test.json",
    ]:
        save_path = os.path.join("inference_res/cache", "{}_{}".format(args.res_flag, file_name))
        if not os.path.exists(save_path):
            continue
        with open(save_path, 'r', encoding='utf-8') as f:
            infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines()) if (line_index - rank) % rank_sum == 0]
        raw_prefixes = [[l['prompt']] for l in infer_data]
        generated_suffixes = [l['infer']["t"] for l in infer_data]

        setup_seed()
        rewards = []
        batch_size = reward_batch_size
        for index in tqdm.tqdm(range(0,len(raw_prefixes), batch_size), desc=f"Rank {rank} rewarding..."):
            if len(raw_prefixes) - index < batch_size:
                batch_size = len(raw_prefixes) - index
            reward_res = get_score(raw_prefixes[index:index+batch_size], generated_suffixes[index:index+batch_size])
            rewards.extend(reward_res.cpu().detach().numpy().tolist())
        assert len(rewards) == len(generated_suffixes) and len(rewards) == len(infer_data), (len(rewards), len(generated_suffixes), len(infer_data))

        for index in range(len(infer_data)):
            infer_data[index]["infer"]["score"] = rewards[index]
        
        save_path = os.path.join("inference_res", "{}_{}".format(args.res_flag, file_name))
        with open(save_path, 'a', encoding='utf-8') as f:
            for line in infer_data:
                content = json.dumps(line, ensure_ascii=False)
                f.write(content+'\n')
    print(f"Rank {rank} completed!")