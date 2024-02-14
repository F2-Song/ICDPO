import tqdm
import sys
sys.path.append("..")
from utils.config import args
if "llama" in args.generator:
    from generator.llama_generator import Llama_Generator
    Generator = Llama_Generator
elif "mistral" in args.generator:
    from generator.mistral_generator import Mistral_Generator
    Generator = Mistral_Generator

def generate(
    data_manager,
):
    generator = Generator(args.model_name_or_path)
    tokenizer = generator.tokenizer
    
    pred_res = []
    batch = []
    if args.batch_size > args.num_generation:
        batch_size = args.batch_size // args.num_generation
    else:
        batch_size = args.batch_size
    pbar = tqdm.tqdm(total=len(data_manager.test_set))
    for sample_index, sample in enumerate(data_manager.test_set):
        batch.append(sample)
        if len(batch) == batch_size or sample_index == len(data_manager.test_set) - 1:
            contexts = []
            raw_texts = []
            for sample in batch:
                context = data_manager.get_context(
                    sample,
                    mode = args.pos_mode,
                    pos_or_neg = args.pos_model_direction,
                    num_demonstrations = args.num_demonstrations,
                )
                contexts.append(context)
                raw_text = data_manager.get_raw_text(sample)
                raw_texts.append(raw_text)
            pred_texts = generator.generate(
                contexts,
                generation_config = {
                    "max_new_tokens": args.max_new_tokens,
                    "num_return_sequences": args.num_generation,
                    "do_sample": args.do_sample,
                }
            ) # [[sent1, sent2, sent3], [sent1, sent2, sent3], ...]
            pred_res.extend(pred_texts)
            pbar.update(len(batch))
            batch = []
    pbar.close()
    return pred_res