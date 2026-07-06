"""
Training script: LoRA fine-tune distilgpt2 on Dolly-15k.

Run with:  python src/train.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from data_pipeline import build_datasets, causal_lm_collate

# ---------------- CONFIG ----------------
# Trained on the full Dolly-15k dataset with LoRA on distilgpt2.
# On CPU this takes several hours; on a free-tier GPU (Colab T4) ~20 min.
CONFIG = {
    "base_model": "distilgpt2",
    "subset_size": None,        # None = use all ~15k Dolly examples
    "max_length": 256,
    "val_frac": 0.08,
    "seed": 42,
    "epochs": 2,
    "batch_size": 4,
    "grad_accum_steps": 1,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["c_attn"],  # GPT-2 style attention projection
    "output_dir": "checkpoints/distilgpt2-dolly-lora",
}
# -----------------------------------------


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    torch.manual_seed(CONFIG["seed"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading and tokenizing Dolly-15k subset...")
    train_data, val_data = build_datasets(
        tokenizer,
        subset_size=CONFIG["subset_size"],
        max_length=CONFIG["max_length"],
        val_frac=CONFIG["val_frac"],
        seed=CONFIG["seed"],
    )
    print(f"Train examples: {len(train_data)} | Val examples: {len(val_data)}")

    # Leakage check (belt-and-suspenders, also covered in tests/)
    train_keys = {tuple(x["input_ids"]) for x in train_data}
    val_keys = {tuple(x["input_ids"]) for x in val_data}
    assert len(train_keys & val_keys) == 0, "Data leakage detected between train and val!"

    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"])
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = lambda batch: causal_lm_collate(batch, tokenizer)



    train_ds = ListDataset(train_data)
    val_ds = ListDataset(val_data)

    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        learning_rate=CONFIG["learning_rate"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final checkpoint...")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"Done. Checkpoint saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
