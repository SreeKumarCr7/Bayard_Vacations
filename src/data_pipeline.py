"""
Data pipeline for Dolly-15k -> prompt/completion format -> tokenized train/val split.

Design decisions (documented per assignment requirement):
- Dataset: databricks/databricks-dolly-15k. Chosen because it's small (13MB, fast to
  download/tokenize), single-turn (no multi-turn flattening logic needed), and
  permissively licensed (CC BY-SA 3.0).
- We drop examples with an empty `response` field (a handful exist) since they'd teach
  the model to predict nothing.
- We truncate to `max_length` tokens (prompt+completion combined) rather than dropping
  long examples outright, to avoid throwing away data.
- Split is done by shuffling indices once with a fixed seed, then slicing — this
  guarantees zero overlap between train/val (tested in tests/test_suite.py).
"""

import random
from datasets import load_dataset


def format_example(example: dict) -> tuple[str, str]:
    """Turn one Dolly row into (prompt, completion) strings."""
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    response = example["response"].strip()

    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return prompt, response


def load_and_format(subset_size: int | None = 4000, seed: int = 42) -> list[dict]:
    """
    Load Dolly-15k, filter empty responses, format, and optionally subsample.

    subset_size=4000 is a deliberate choice: enough to show a real training curve
    within a couple of hours on a free-tier GPU, small enough to not blow the time box.
    Set to None to use the full 15k.
    """
    raw = load_dataset("databricks/databricks-dolly-15k", split="train")

    examples = []
    for row in raw:
        if not row["response"] or not row["response"].strip():
            continue  # filtering decision: drop empty responses
        prompt, completion = format_example(row)
        examples.append({"prompt": prompt, "completion": completion})

    rng = random.Random(seed)
    rng.shuffle(examples)

    if subset_size is not None:
        examples = examples[:subset_size]

    return examples


def train_val_split(examples: list[dict], val_frac: float = 0.05, seed: int = 42):
    """Shuffle once, slice once -> guarantees no leakage between splits."""
    rng = random.Random(seed)
    shuffled = examples[:]  # examples already shuffled in load_and_format, but be explicit
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_frac))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def tokenize_example(example: dict, tokenizer, max_length: int = 512) -> dict:
    """
    Tokenize prompt+completion together for causal LM training.
    Labels = input_ids, but we mask the prompt tokens with -100 so loss is only
    computed on the completion (standard instruction-tuning practice).
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        example["completion"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + completion_ids
    input_ids = input_ids[:max_length]

    labels = [-100] * len(prompt_ids) + completion_ids
    labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_datasets(tokenizer, subset_size: int | None = 4000, max_length: int = 512,
                    val_frac: float = 0.05, seed: int = 42):
    """Full pipeline: load -> filter/format -> split -> tokenize. Returns (train, val) as lists of dicts."""
    examples = load_and_format(subset_size=subset_size, seed=seed)
    train_raw, val_raw = train_val_split(examples, val_frac=val_frac, seed=seed)

    train_tok = [tokenize_example(ex, tokenizer, max_length) for ex in train_raw]
    val_tok = [tokenize_example(ex, tokenizer, max_length) for ex in val_raw]

    return train_tok, val_tok

def causal_lm_collate(batch: list[dict], tokenizer):
    import torch
    max_len = max(len(ex["input_ids"]) for ex in batch)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }