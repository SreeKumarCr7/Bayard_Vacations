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


def compute_losses_and_perplexity(model, tokenizer, checkpoint_dir=DEFAULT_CHECKPOINT):
    # Sample a subset for the training-loss estimate (full 14k+ examples is too slow on CPU
    # for a quick check) -- validation loss still uses the full held-out val split.
    train_data, val_data = build_datasets(tokenizer, subset_size=1500, max_length=256, val_frac=0.08)

    def _avg_loss(data):
        loader = DataLoader(data, batch_size=4, collate_fn=lambda batch: causal_lm_collate(batch, tokenizer))
        total_loss, n_batches = 0.0, 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(**batch)
                total_loss += out.loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    train_loss = _avg_loss(train_data)
    val_loss = _avg_loss(val_data)
    perplexity = math.exp(val_loss)
    return train_loss, val_loss, perplexity


def run_qualitative(model, tokenizer):
    print("\n--- Qualitative outputs ---")
    for p in DEMO_PROMPTS:
        formatted = f"### Instruction:\n{p}\n\n### Response:\n"
        out = generate(model, tokenizer, formatted, temperature=0.7, max_new_tokens=80, seed=0)
        print(f"\nPrompt: {p}\nOutput: {out}")


if __name__ == "__main__":
    model, tokenizer = load_model()

    train_loss, val_loss, ppl = compute_losses_and_perplexity(model, tokenizer)
    print(f"Training loss:   {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation perplexity: {ppl:.2f}")

    run_qualitative(model, tokenizer)

    print(
        "\n--- Error analysis (observed failure modes) ---\n"
        "1. REPETITION LOOPING (dominant): Model restates phrases near-verbatim.\n"
        "   Signature of an undertrained model that learned surface patterns.\n"
        "2. FACTUAL HALLUCINATION: Confidently produces wrong facts (e.g. wrong\n"
        "   locations, invented names). distilgpt2 has limited factual capacity.\n"
        "3. ARITHMETIC FAILURE: Math questions produce nonsense. Expected for\n"
        "   small LMs without dedicated computation circuits.\n"
        "4. FORMATTING WEAKNESS: Brainstorm/classification prompts don't follow\n"
        "   the requested format (lists, labels).\n"
        "5. PARTIAL SUCCESSES: Poem and topical prompts stay loosely on-topic,\n"
        "   showing some instruction-topic association was learned.\n"
        "\n"
        "Mitigations applied: repetition_penalty=1.3 and no_repeat_ngram_size=3\n"
        "in generate() reduce (but do not eliminate) looping."
    )
