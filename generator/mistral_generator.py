import os
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
import sys
sys.path.append("..")
from modeling.modeling_mistral import MistralForCausalLM
import torch
import tqdm

class Mistral_Generator:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": self.tokenizer.eos_token,
            }
        )
        # use bf16
        self.model = MistralForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to("cuda:0").eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        self.custom_generation_config = {
            "top_p": 0.8,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
    
    def batch_generation(self, batch_text, other_config):
        original_padding_side, original_truncation_side = self.tokenizer.padding_side, self.tokenizer.truncation_side
        self.tokenizer.padding_side, self.tokenizer.truncation_side = "left", "left"

        batch_inputs = self.tokenizer(batch_text, padding=True, return_tensors="pt")
        batch_inputs = {key: batch_inputs[key].to(self.model.device) for key in batch_inputs if key != "token_type_ids"}
        with torch.no_grad():
            batch_pred = self.model.generate(
                **batch_inputs,
                **other_config,
            )
        self.tokenizer.padding_side, self.tokenizer.truncation_side = original_padding_side, original_truncation_side

        return batch_pred.cpu().detach().numpy().tolist()
    
    def postpreprocess(self, preds, contexts=None):
        assert len(preds) % len(contexts) == 0
        num_generation = len(preds) // len(contexts)

        preds = self.tokenizer.batch_decode(
            preds, 
            skip_special_tokens=True
        )
        preds = [preds[index:index+num_generation] for index in range(0, len(preds), num_generation)]
        pred_texts = []
        for context, texts in zip(contexts, preds):
            batch_pred_texts = []
            for text in texts:
                assert context == text[:len(context)]
                batch_pred_texts.append(text[len(context):])
            pred_texts.append(batch_pred_texts)
        return pred_texts
    
    def generate(self, contexts, generation_config):
        custom_generation_config = {
            **self.custom_generation_config,
        }
        custom_generation_config.update(generation_config)

        preds = self.batch_generation(
            contexts, 
            other_config = custom_generation_config,
        )
        
        pred_texts = self.postpreprocess(preds, contexts)
        return pred_texts