#!/usr/bin/env python
# train_sft.py
# Launch with: accelerate launch train_sft.py

import os
import random
import torch

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from tqdm import tqdm

# — GPU configuration —
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# — Fixed hyperparameters — 
SEED            = 42
# DATA_PATH       = "../datasets/cqa_sft_prompt_completion.jsonl"
DATA_PATH       = "../datasets/cqa_full_training_prompt_completion.jsonl"
SEQ_LENGTH      = 1792

# — SFT hyperparameters —
NUM_EPOCHS      = 1
SAVE_STEPS      = 1     
LOG_STEPS       = 2     
EVAL_STEPS      = 2     
BATCH_SIZE      = 1
EVAL_BATCH      = 1
ACCUM_STEPS     = 4
LR              = 1e-4
LR_SCHED        = "cosine"
WEIGHT_DECAY    = 0.05
OPTIM           = "paged_adamw_32bit"
BF16            = True
GRAD_CHECKPOINT = False
GROUP_BY_LENGTH = False
REPORT_TO       = ["tensorboard"]

# — Models to fine-tune —
MODEL_NAMES = {
    "Meta-Llama-3.1-8B": "../model_cache/Meta-Llama-3.1-8B",
    "Aya-23-8B":         "../model_cache/Aya-23-8B",
    "SeaLLMs-v3-7B":     "../model_cache/SeaLLMs-v3-7B",
    "SEA-LION-v3-8B":    "../model_cache/SEA-LION-v3-8B",
    "Sahabat-AI-8B":     "../model_cache/Sahabat-AI-8B"
}

def create_datasets(tokenizer):
    # non-streaming split 90% train / 10% valid
    raw = load_dataset("json", data_files=DATA_PATH, split="train", streaming=False)
    parts = raw.train_test_split(test_size=0.1, seed=SEED)
    train_raw, valid_raw = parts["train"], parts["test"]

    # compute chars_per_token (optional; TRL can auto-estimate if None)
    total_chars = total_tokens = 0
    for ex in tqdm(train_raw, desc="Estimating ratio", total=len(train_raw)):
        text = ex["prompt"] + ex["completion"]
        total_chars += len(text)
        total_tokens += len(tokenizer(text).tokens()) if tokenizer.is_fast else len(tokenizer.tokenize(text))
    chars_ratio = total_chars / total_tokens
    print(f"chars/token ≃ {chars_ratio:.2f}")
    
    max_p, max_c, max_pc = 0, 0, 0
    for ex in tqdm(train_raw, desc="Counting token lengths", total=len(train_raw)):
        # tanpa special tokens
        p_len = len(tokenizer(ex["prompt"], add_special_tokens=False).input_ids)
        c_len = len(tokenizer(ex["completion"], add_special_tokens=False).input_ids)
        pc_len = len(tokenizer(ex["prompt"] + ex["completion"], add_special_tokens=False).input_ids)
        max_p = max(max_p, p_len)
        max_c = max(max_c, c_len)
        max_pc = max(max_pc, pc_len)
    print(f"Max prompt tokens: {max_p}")
    print(f"Max completion tokens: {max_c}")
    print(f"Max prompt+completion tokens: {max_pc}")

    # wrap into ConstantLengthDataset
    train_ds = ConstantLengthDataset(
        tokenizer, train_raw,
        formatting_func=lambda ex: ex["prompt"] + ex["completion"],
        infinite=False,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_ratio
    )
    valid_ds = ConstantLengthDataset(
        tokenizer, valid_raw,
        formatting_func=lambda ex: ex["prompt"] + ex["completion"],
        infinite=False,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_ratio
    )
    return train_ds, valid_ds


def main():
    set_seed(SEED)
    random.seed(SEED)
    device_idx = PartialState().process_index

    for key, model_path in MODEL_NAMES.items():
        print(f"\n=== Fine-tuning {key} ===")

        # 1) Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="right", local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 2) Quantization config
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 3) Load base model
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map={"": device_idx},
            trust_remote_code=True,
            local_files_only=True
        )
        base_model.config.use_cache = False

        # 4) Prepare LoRA config (trainer will wrap)
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj","v_proj"],
            bias="none"
        )

        # 5) Create train & valid datasets
        train_ds, valid_ds = create_datasets(tokenizer)

        # 6) Prepare SFTConfig & dirs
        output_dir = f"sft_output_{key}"
        os.makedirs(output_dir, exist_ok=True)
        training_args = SFTConfig(
            output_dir               = output_dir,
            num_train_epochs         = NUM_EPOCHS,
            save_strategy            = "steps",
            save_steps               = SAVE_STEPS,
            logging_strategy         = "steps",
            logging_steps            = LOG_STEPS,
            eval_strategy            = "steps",
            eval_steps               = EVAL_STEPS,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = EVAL_BATCH,
            gradient_accumulation_steps = ACCUM_STEPS,
            learning_rate            = LR,
            lr_scheduler_type        = LR_SCHED,
            weight_decay             = WEIGHT_DECAY,
            optim                    = OPTIM,
            bf16                     = BF16,
            group_by_length          = GROUP_BY_LENGTH,
            gradient_checkpointing   = GRAD_CHECKPOINT,
            report_to                = REPORT_TO,
            max_length               = None,
            logging_first_step = True,
            do_train = True,
            do_eval = True
        )

        # 7) Initialize SFTTrainer
        trainer = SFTTrainer(
            model           = base_model,
            train_dataset   = train_ds,
            eval_dataset    = valid_ds,
            args            = training_args,
            peft_config     = peft_cfg,
            processing_class= tokenizer
        )

        # 8) Train!
        trainer.train()

        # 9) Save trainer checkpoint + adapter config
        trainer.save_model(output_dir)

        # 10) Save PEFT-wrapped model
        ckpt = os.path.join(output_dir, "final_checkpoint")
        trainer.model.save_pretrained(ckpt)

        # 11) Merge adapters & save final
        del base_model, trainer
        
        print(f"✅ Selesai fine-tuning dan menyimpan semua checkpoint untuk {key}")

        # final cleanup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
