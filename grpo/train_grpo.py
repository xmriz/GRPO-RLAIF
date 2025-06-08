#!/usr/bin/env python
# train_grpo.py
# Run with: accelerate launch train_grpo.py

import os
import random
import torch
import re

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer

# — GPU configuration —
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# — Fixed hyperparameters —
SEED                         = 42
DATA_PATH                    = "../datasets/cqa_full_training_prompt_completion.jsonl"
NUM_EPOCHS                   = 1
PER_DEVICE_BATCH             = 1
EVAL_BATCH                   = 2
NUM_GENERATIONS              = 2
GRADIENT_ACCUMULATION_STEPS  = 2
MAX_PROMPT_LENGTH            = 1792
MAX_COMPLETION_LEN           = 256
LEARNING_RATE                = 1e-6
EPSILON                      = 0.2
BETA                         = 0.04
LOSS_TYPE                    = "dr_grpo"
SCALE_REWARDS                = False
LOGGING_STEPS                = 10
SAVE_STEPS                   = 300
SAVE_TOTAL_LIMIT             = 3
TEMPERATURE                  = 0.9
TOP_P                        = 1.0
TOP_K                        = 50
REPETITION_PENALTY           = 1.0
GRADIENT_CHECKPOINT          = False
EVAL_STEPS                   = 500

#  — Paths —
# 1) Base model cache (full LLM with vocab/config)
BASE_MODEL_PATH = "../model_cache/SeaLLMs-v3-7B"
# 2) LoRA adapter checkpoint from SFT
ADAPTER_PATH    = "../sft/sft_output_SeaLLMs-v3-7B/final_checkpoint"

#  If you have multiple SFT LoRA checkpoints, add in similar fashion:
MODEL_NAMES = {
    "SeaLLMs-v3-7B-SFT": ADAPTER_PATH,
}

def main():
    # 1) Set global seed
    set_seed(SEED)
    random.seed(SEED)

    # 2) Load dataset JSONL and rename column 'completion' → 'answer'
    raw = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train",
        streaming=False
    )
    print("Kolom sebelum rename:", raw.column_names)  # Expect ['prompt','completion']
    raw = raw.rename_column("completion", "answer")
    print("Kolom sesudah rename:", raw.column_names)  # Now ['prompt','answer']

    # 3) 90% train / 10% eval
    split_datasets = raw.train_test_split(test_size=0.10, seed=SEED)
    train_dataset   = split_datasets["train"]
    eval_dataset    = split_datasets["test"]
    print(f"→ Total contoh: {len(raw)}")
    print(f"→ Train contoh:  {len(train_dataset)}")
    print(f"→ Eval contoh:   {len(eval_dataset)}")

    # 4) Determine local GPU index for Accelerate
    device_idx = PartialState().process_index

    for model_key, adapter_dir in MODEL_NAMES.items():
        print(f"\n=== Training GRPO: {model_key} (GPU idx {device_idx}) ===")

        # 5) Load tokenizer from base model cache
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,        # <-- note: base, not adapter
            padding_side="left",
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 6) 4-bit quantization config
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit           = True,
            bnb_4bit_quant_type    = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
        )

        # 7) Load base model (4-bit) from BASE_MODEL_PATH
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config = bnb_cfg,
            device_map          = {"": device_idx},
            trust_remote_code   = True,
            local_files_only    = True
        )
        base_model.config.use_cache = False

        # 8) Wrap with LoRA adapter from SFT checkpoint (adapter_dir)
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,            # <-- LoRA adapter checkpoint
            torch_dtype=torch.bfloat16,
            is_trainable=True,      # ensure LoRA is trainable for GRPO
            local_files_only=True
        )
        model.config.use_cache = False

        # 9) Define reward function using renamed column 'answer'
        def reward_cqa(prompts=None, completions=None, answer=None, **kwargs):
            """
            Reward function:
            - completions: List[str] with generated texts (decoded)
            - answer:      List[str] ground-truth answers (column 'answer')
            """
            responses = completions or []
            gt_list   = answer      or []

            rewards = []
            for idx, (gen_text, gt_text) in enumerate(zip(responses, gt_list)):
                gt_strip  = gt_text.strip()
                gen_strip = gen_text.strip()

                # Debug prints for first 3 in batch
                if idx < 3:
                    print(f"  [DEBUG Sample {idx}] gt => \"{gt_strip}\"")
                    print(f"  [DEBUG Sample {idx}] gen=> \"{gen_strip}\"")

                # 1) Unanswerable marker “Saya tidak tahu”
                if gt_strip.startswith("Saya tidak tahu"):
                    reward = 0.5 if "Saya tidak tahu" in gen_strip else 0.0
                    if idx < 3:
                        print(f"    -> Saw 'Saya tidak tahu': reward = {reward}")
                    rewards.append(reward)
                else:
                    # 2) Overlap token score (case-insensitive)
                    tokens_gt  = re.findall(r"\w+", gt_strip.lower())
                    tokens_gen = re.findall(r"\w+", gen_strip.lower())
                    if idx < 3:
                        print(f"    tokens_gt  (len {len(tokens_gt)}): {tokens_gt}")
                        print(f"    tokens_gen (len {len(tokens_gen)}): {tokens_gen}")

                    if not tokens_gt:
                        if idx < 3:
                            print("    -> gt tokens kosong, reward = 0.0")
                        rewards.append(0.0)
                    else:
                        common = set(tokens_gt) & set(tokens_gen)
                        score  = len(common) / len(tokens_gt)
                        if idx < 3:
                            print(f"    -> common tokens: {common}, score={score:.4f}")
                        rewards.append(float(score))

            if len(rewards) <= 5:
                print("  - rewards:", rewards)
            else:
                print("  - rewards (first 5):", rewards[:5])
            return rewards

        # 10) Prepare GRPOConfig and output directory
        output_dir = f"grpo_output_{model_key}"
        os.makedirs(output_dir, exist_ok=True)
        training_args = GRPOConfig(
            output_dir                   = output_dir,
            overwrite_output_dir         = True,

            # — Training & Eval —
            do_train                     = True,
            do_eval                      = True,
            num_train_epochs             = NUM_EPOCHS,
            per_device_train_batch_size  = PER_DEVICE_BATCH,
            per_device_eval_batch_size   = EVAL_BATCH,
            gradient_accumulation_steps  = GRADIENT_ACCUMULATION_STEPS,

            learning_rate                = LEARNING_RATE,
            lr_scheduler_type            = "linear",
            warmup_ratio                 = 0.0,

            # — Data preprocessing —
            remove_unused_columns        = False,
            label_names                  = ["answer"],  # <-- renamed column
            max_prompt_length            = MAX_PROMPT_LENGTH,

            # — Generation during GRPO —
            num_generations              = NUM_GENERATIONS,
            max_completion_length        = MAX_COMPLETION_LEN,
            temperature                  = TEMPERATURE,
            top_p                        = TOP_P,
            top_k                        = TOP_K,
            repetition_penalty           = REPETITION_PENALTY,

            # — Loss hyperparameters (Dr. GRPO) —
            epsilon                      = EPSILON,
            beta                         = BETA,
            loss_type                    = LOSS_TYPE,
            scale_rewards                = SCALE_REWARDS,
            mask_truncated_completions   = False,

            # — Logging & Saving —
            logging_dir                  = os.path.join(output_dir, "logs"),
            logging_strategy             = "steps",
            logging_steps                = LOGGING_STEPS,
            save_strategy                = "steps",
            save_steps                   = SAVE_STEPS,
            save_total_limit             = SAVE_TOTAL_LIMIT,
            report_to                    = ["tensorboard"],

            # — Evaluation —
            eval_strategy                = "steps",
            eval_steps                   = EVAL_STEPS,

            # — Gradient Checkpointing —
            gradient_checkpointing       = GRADIENT_CHECKPOINT,
        )

        # 11) Initialize GRPOTrainer
        trainer = GRPOTrainer(
            model                      = model,
            args                       = training_args,
            train_dataset              = train_dataset,
            eval_dataset               = eval_dataset,
            processing_class           = tokenizer,        # tokenizer to process inputs
            reward_funcs               = [reward_cqa],     # list of reward functions
            reward_processing_classes  = [None],            # ground truth already string
        )

        # 12) Start training
        trainer.train()

        # 13) Save checkpoint & LoRA adapter
        trainer.save_model(output_dir)

        # 14) Save final LoRA-wrapped model for downstream use
        final_ckpt = os.path.join(output_dir, "final_checkpoint")
        trainer.model.save_pretrained(final_ckpt)

        # 15) Cleanup
        del base_model, model, trainer
        torch.cuda.empty_cache()

        print(f"✅ GRPO selesai untuk {model_key}. Checkpoint akhir di: {final_ckpt}")

if __name__ == "__main__":
    main()
