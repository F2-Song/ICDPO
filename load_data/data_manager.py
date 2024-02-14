import sys
sys.path.append("..")
import load_data.hh as hh
import load_data.syntheticgpt as SyntheticGPT
from utils.utils import load_raw_dataset, save_dataset

class Data_Manager():
    def __init__(self, path, task):
        self.test_set = load_raw_dataset(path)
        if "hh" in task:
            self.parser = hh
        elif "SyntheticGPT" in task:
            self.parser = SyntheticGPT

    def get_context(
        self,
        sample,
        mode,
        pos_or_neg="pos",
        num_demonstrations=3,
    ):
        if mode == "base" or mode == "sft":
            return self.parser.get_base_context(sample)
        elif mode == "icl":
            return self.parser.get_icl_context(sample, num_demonstrations, pos_or_neg)
        else:
            raise Exception("Invalid mode!")
    
    def get_raw_text(self, sample):
        return self.parser.get_raw_text(sample)
    
    def postpreprocess(
        self, 
        sample,
        do_early_truncation=True,
    ):
        return self.parser.postpreprocess(
            sample,
            do_early_truncation=do_early_truncation,
        )

    def save_test_set(self, path):
        save_dataset(self.test_set, path)
        return True
        