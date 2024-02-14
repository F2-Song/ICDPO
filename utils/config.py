import random
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ICDPO")
    parser.add_argument(
        "--index", 
        type=str, 
        default="0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hh",
    )
    parser.add_argument(
        "--generator", 
        type=str,
    )
    parser.add_argument(
        "--retrieval",
        type=str,
    )
    parser.add_argument(
        "--pos_mode",
        type=str,
        help="Can be icl, base or sft",
    )
    parser.add_argument(
        "--neg_mode",
        type=str,
        help="Can be icl, base or sft",
    )
    parser.add_argument(
        "--pos_model_direction",
        type=str,
        default="pos", # pos or neg
    )
    parser.add_argument(
        "--neg_model_direction",
        type=str,
        default="neg", # pos or neg
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
    )
    parser.add_argument(
        "--num_demonstrations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_generation",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

args = parse_args()
setup_seed(args.seed)
args_message = '\n'+'\n'.join([f'{k:<40}: {v}' for k, v in vars(args).items()])
print(args_message)