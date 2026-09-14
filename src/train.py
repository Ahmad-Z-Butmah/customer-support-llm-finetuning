import os
from pathlib import Path

import torch
import wandb

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig

from trl import (
    SFTConfig,
    SFTTrainer,
)


load_dotenv()


BASE_MODEL = "meta-llama/Llama-3.2-3B"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "models/customer-support-llama"

MAX_SEQUENCE_LENGTH = 256

EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. "
            "Run training on Google Colab with a T4 GPU."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN missing."
        )

    login(
        token=hf_token,
        add_to_git_credential=True,
    )

    wandb_key = os.getenv(
        "WANDB_API_KEY"
    )

    use_wandb = bool(wandb_key)

    if use_wandb:
        wandb.login(
            key=wandb_key
        )

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALIDATION_FILE,
        },
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset[
        "validation"
    ]

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "right"

    capability = (
        torch.cuda
        .get_device_capability()
    )

    use_bf16 = capability[0] >= 8

    print(
        "Using BF16:",
        use_bf16,
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if use_bf16
                else torch.float16
            ),
        )
    )

    print(
        "Loading Llama 3.2 3B..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
        )
    )

    model.config.use_cache = False

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )

    print(
        "Model memory:",
        f"{model.get_memory_footprint() / 1e9:.2f} GB",
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    training_config = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=(
            BATCH_SIZE
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        optim="paged_adamw_32bit",

        warmup_ratio=0.03,

        lr_scheduler_type="cosine",

        weight_decay=0.001,

        fp16=not use_bf16,
        bf16=use_bf16,

        max_grad_norm=0.3,

        logging_steps=10,

        eval_strategy="steps",
        eval_steps=50,

        save_strategy="steps",
        save_steps=50,

        save_total_limit=2,

        max_length=(
            MAX_SEQUENCE_LENGTH
        ),

        report_to=(
            "wandb"
            if use_wandb
            else "none"
        ),

        run_name=(
            "customer-support-qlora"
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            validation_dataset
        ),

        peft_config=lora_config,
    )

    print()
    print(
        "Starting QLoRA training..."
    )

    trainer.train()

    Path(
        OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print()
    print(
        f"Model saved to {OUTPUT_DIR}"
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()