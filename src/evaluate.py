"""
Evaluation: quantitative (val loss + perplexity) and qualitative (sample generations).

Usage: python src/evaluate.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import math
import torch
from torch.utils.data import DataLoader


from generate import load_model, generate, DEFAULT_CHECKPOINT
from data_pipeline import build_datasets, causal_lm_collate

# Representative held-out prompts spanning Dolly's task categories
DEMO_PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Classify the sentiment of this sentence: 'I loved the movie.'",
    "Brainstorm three ideas for a birthday party theme.",
    "Explain what machine learning is to a five-year-old.",
    "What is 15 multiplied by 7?",
    "Give me a recipe idea using chicken and rice.",
]


def compute_val_loss_and_perplexity(model, tokenizer, checkpoint_dir=DEFAULT_CHECKPOINT):
    _, val_data = build_datasets(tokenizer, subset_size=600, max_length=256, val_frac=0.08)

    
    loader = DataLoader(val_data, batch_size=4, collate_fn=lambda batch: causal_lm_collate(batch, tokenizer))

    total_loss, n_batches = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def run_qualitative(model, tokenizer):
    print("\n--- Qualitative outputs ---")
    for p in DEMO_PROMPTS:
        formatted = f"### Instruction:\n{p}\n\n### Response:\n"
        out = generate(model, tokenizer, formatted, temperature=0.7, max_new_tokens=80, seed=0)
        print(f"\nPrompt: {p}\nOutput: {out}")


if __name__ == "__main__":
    model, tokenizer = load_model()

    val_loss, ppl = compute_val_loss_and_perplexity(model, tokenizer)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation perplexity: {ppl:.2f}")

    run_qualitative(model, tokenizer)

    print(
        "\n--- Error analysis (fill in after reviewing outputs) ---\n"
        "Expected failure modes for a distilgpt2-scale LoRA fine-tune on ~4k examples:\n"
        "- Factual recall (capitals, dates, numeric facts) will often be wrong or made up.\n"
        "- Arithmetic will usually fail outright.\n"
        "- Long-range coherence over >2-3 sentences will degrade.\n"
        "- Category-specific formatting (e.g. brainstorm lists) may not be followed reliably.\n"
        "Note which of the 8 prompts above actually exhibited these failures."
    )
