from utils.config import args
from load_data.data_manager import Data_Manager
from utils.generation_utils import generate

path = "data/"
if "hh" in args.task:
    task = args.task.replace("_harmless_base", "").replace("_helpful_base", "")
    path += "{}_test_{}_demos/".format(task, args.retrieval)
    if "harmless_base" in args.task:
        path += "harmless_base.json"
    elif "helpful_base" in args.task:
        path += "helpful_base.json"
    else:
        raise Exception("Invalid task!")
else:
    assert "SytheticGPT" in args.task
    path += "{}_test_{}_demos/test.json".format(args.task, args.retrieval)

data_manager = Data_Manager(path, args.task)
pred_res = generate(data_manager)
for sample, candidates in zip(data_manager.test_set, pred_res):
    sample["candidates"] = []
    for candidate in candidates:
        sample["candidates"].append({"t": candidate})
data_manager.save_test_set(f"inference_res/cache/{args.output_dir}")