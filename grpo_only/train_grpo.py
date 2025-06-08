#!/usr/bin/env python
# train_grpo.py
# Run with: accelerate launch train_grpo.py

import os
import random
import torch
import re

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model
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

MODEL_NAMES = {
    "SeaLLMs-v3-7B": "../model_cache/SeaLLMs-v3-7B",
}

def main():
    # 1) Set seed
    set_seed(SEED)
    random.seed(SEED)

    # 2) Load and rename 'completion' → 'answer'
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

    print("Jumlah contoh di train_dataset:", len(train_dataset))
    print("Contoh pertama:\n", train_dataset[0])
    print("\nLima contoh pertama:")
    for i in range(5):
        print(f"[{i}]", train_dataset[i])

    print(f"→ Total contoh: {len(raw)}")
    print(f"→ Train contoh:  {len(train_dataset)}")
    print(f"→ Eval contoh:   {len(eval_dataset)}")

    # 4) Determine local GPU index
    device_idx = PartialState().process_index

    for model_key, model_path in MODEL_NAMES.items():
        print(f"\n=== Training GRPO: {model_key} (GPU idx {device_idx}) ===")

        # 5) Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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

        # 7) Load the base model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config = bnb_cfg,
            device_map          = {"": device_idx},
            trust_remote_code   = True,
            local_files_only    = True
        )
        base_model.config.use_cache = False

        # 8) Wrap with LoRA
        peft_cfg = LoraConfig(
            task_type      = "CAUSAL_LM",
            r              = 4,
            lora_alpha     = 16,
            lora_dropout   = 0.05,
            target_modules = ["q_proj", "v_proj"],
            bias           = "none",
        )
        model = get_peft_model(base_model, peft_cfg)

        # -------------------------------------------------------
        # 9) Define the reward function that expects `answer` instead of `completion`
        # -------------------------------------------------------
        def reward_cqa(prompts=None, completions=None, answer=None, **kwargs):
            """
            Reward function:
            - prompts    : List[str] — original prompts
            - completions: List[str] — generated candidate texts
            - answer     : List[str] — ground-truth answers (renamed from 'completion')
            - **kwargs   : any other fields (not used here)
            """
            responses = completions or []
            gt_list   = answer      or []

            rewards = []
            for idx, (gen_text, gt_text) in enumerate(zip(responses, gt_list)):
                gt_strip  = gt_text.strip()
                gen_strip = gen_text.strip()

                if idx < 3:
                    print(f"  [DEBUG Sample {idx}] gt_strip  ==> \"{gt_strip}\"")
                    print(f"  [DEBUG Sample {idx}] gen_strip ==> \"{gen_strip}\"")

                if gt_strip.startswith("Saya tidak tahu"):
                    reward = 0.5 if "Saya tidak tahu" in gen_strip else 0.0
                    if idx < 3:
                        print(f"    -> Branch ‘Saya tidak tahu’: reward = {reward}")
                    rewards.append(reward)
                else:
                    tokens_gt  = re.findall(r"\w+", gt_strip.lower())
                    tokens_gen = re.findall(r"\w+", gen_strip.lower())
                    if idx < 3:
                        print(f"    tokens_gt  (len {len(tokens_gt)}): {tokens_gt}")
                        print(f"    tokens_gen (len {len(tokens_gen)}): {tokens_gen}")
                    if not tokens_gt:
                        if idx < 3:
                            print("    -> tokens_gt kosong, reward = 0.0")
                        rewards.append(0.0)
                    else:
                        common = set(tokens_gt) & set(tokens_gen)
                        score  = len(common) / len(tokens_gt)
                        if idx < 3:
                            print(f"    -> common tokens: {common}, score = {score:.4f}")
                        rewards.append(float(score))

            if len(rewards) <= 5:
                print("  - rewards keseluruhan:", rewards)
            else:
                print("  - rewards (contoh 5 pertama):", rewards[:5])

            return rewards
        # -------------------------------------------------------
        # End of reward_cqa
        # -------------------------------------------------------

        # 10) Create output directory & GRPOConfig
        output_dir = f"grpo_output_{model_key.replace('/', '_')}"
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
            label_names                  = ["answer"],   # <-- use the renamed column here
            max_prompt_length            = MAX_PROMPT_LENGTH,

            # — Generation during GRPO —
            num_generations              = NUM_GENERATIONS,
            max_completion_length        = MAX_COMPLETION_LEN,
            temperature                  = TEMPERATURE,
            top_p                        = TOP_P,
            top_k                        = TOP_K,
            repetition_penalty           = REPETITION_PENALTY,

            # — Loss hyperparameters —
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

        # 11) Initialize GRPOTrainer with `reward_cqa` and no extra tokenizers
        trainer = GRPOTrainer(
            model                      = model,
            args                       = training_args,
            train_dataset              = train_dataset,
            eval_dataset               = eval_dataset,
            processing_class           = tokenizer,
            reward_funcs               = [reward_cqa],  # must be a list
            reward_processing_classes  = [None],        # ground truth is already a string
        )

        # 12) Start training
        trainer.train()

        # 13) Save checkpoint & LoRA adapter
        trainer.save_model(output_dir)

        # 14) Save final LoRA-wrapped model
        ckpt = os.path.join(output_dir, "final_checkpoint")
        trainer.model.save_pretrained(ckpt)

        # 15) Free memory
        del base_model, trainer
        torch.cuda.empty_cache()
        print(f"✅ GRPO selesai untuk {model_key}. Checkpoint akhir di: {ckpt}")

if __name__ == "__main__":
    main()
