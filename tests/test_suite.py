"""
Test suite covering all 10 required scenarios from the interview guide.

Run with: pytest tests/ -v

Note: tests 3-4 and 8-10 need a trained/loaded checkpoint. If checkpoints/ doesn't
exist yet, those tests are skipped rather than failed (see `_checkpoint_available`)
so the full suite is still runnable before training finishes.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_pipeline import load_and_format, train_val_split, tokenize_example, build_datasets
from generate import DEFAULT_CHECKPOINT, load_model, generate

BASE_MODEL = "distilgpt2"


def _checkpoint_available():
    return os.path.isdir(DEFAULT_CHECKPOINT) and len(os.listdir(DEFAULT_CHECKPOINT)) > 0


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------- 1. Tokenizer round-trip ----------
def test_tokenizer_roundtrip(tokenizer):
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    decoded = tokenizer.decode(ids)
    assert decoded.strip() == text.strip()


# ---------- 2. Dataset loader: shapes, split sizes, no leakage ----------
def test_dataset_split_no_leakage(tokenizer):
    train, val = build_datasets(tokenizer, subset_size=200, max_length=128, val_frac=0.1)

    assert len(train) + len(val) == 200

    train_prompts = {tuple(t["input_ids"]) for t in train}
    val_prompts = {tuple(v["input_ids"]) for v in val}
    assert len(train_prompts & val_prompts) == 0

    for ex in train[:5]:
        assert isinstance(ex["input_ids"], list)
        assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


# ---------- 3. Overfit sanity check ----------
@pytest.mark.slow
def test_overfit_small_batch(tokenizer):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    examples = load_and_format(subset_size=12, seed=1)
    tokenized = [tokenize_example(ex, tokenizer, max_length=64) for ex in examples]

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(tokenized, batch_size=4, collate_fn=collator, shuffle=True)

    model.train()
    losses = []
    for epoch in range(30):
        for batch in loader:
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())

    assert losses[-1] < losses[0] * 0.3, f"Loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"


# ---------- 4. Post-training generation produces non-trivial output ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_generation_non_trivial():
    model, tokenizer = load_model()
    prompts = [
        "### Instruction:\nWhat is the capital of Japan?\n\n### Response:\n",
        "### Instruction:\nWrite a sentence about the weather.\n\n### Response:\n",
        "### Instruction:\nName a fruit.\n\n### Response:\n",
    ]
    for p in prompts:
        out = generate(model, tokenizer, p, temperature=0.8, max_new_tokens=30, seed=1)
        completion = out[len(p):].strip() if out.startswith(p) else out.strip()
        assert len(completion) > 0
        tokens = completion.split()
        if len(tokens) > 3:
            assert len(set(tokens)) > 1, "Output is a single repeated token"


# ---------- 5. Empty prompt doesn't crash ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_empty_prompt_no_crash():
    model, tokenizer = load_model()
    out = generate(model, tokenizer, "", max_new_tokens=10)
    assert isinstance(out, str)


# ---------- 6. Overlong prompt truncated gracefully ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_overlong_prompt_truncated():
    model, tokenizer = load_model()
    huge_prompt = "word " * 5000
    out = generate(model, tokenizer, huge_prompt, max_new_tokens=10)
    assert isinstance(out, str)


# ---------- 7. max_new_tokens edge cases ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_max_new_tokens_edge_cases():
    model, tokenizer = load_model()

    out_zero = generate(model, tokenizer, "Hello", max_new_tokens=0)
    assert isinstance(out_zero, str)

    out_large = generate(model, tokenizer, "Hello", max_new_tokens=100_000)
    # enforced cap in generate() prevents runaway generation
    ids = tokenizer(out_large)["input_ids"]
    assert len(ids) < 100_000


# ---------- 8. Greedy decoding determinism ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_greedy_determinism():
    model, tokenizer = load_model()
    prompt = "### Instruction:\nWhat is 2 plus 2?\n\n### Response:\n"
    out1 = generate(model, tokenizer, prompt, do_sample=False, max_new_tokens=20, seed=42)
    out2 = generate(model, tokenizer, prompt, do_sample=False, max_new_tokens=20, seed=42)
    assert out1 == out2


# ---------- 9. Sampling seeded correctly ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_sampling_seeding():
    model, tokenizer = load_model()
    prompt = "### Instruction:\nTell me something interesting.\n\n### Response:\n"

    out_a1 = generate(model, tokenizer, prompt, do_sample=True, temperature=1.0, max_new_tokens=30, seed=1)
    out_a2 = generate(model, tokenizer, prompt, do_sample=True, temperature=1.0, max_new_tokens=30, seed=1)
    assert out_a1 == out_a2  # same seed -> same output

    out_b = generate(model, tokenizer, prompt, do_sample=True, temperature=1.0, max_new_tokens=30, seed=2)
    assert out_a1 != out_b  # different seed -> (almost certainly) different output


# ---------- 10. Fixed prompt set quality gate ----------
@pytest.mark.skipif(not _checkpoint_available(), reason="no trained checkpoint yet")
def test_fixed_prompt_quality_gate():
    model, tokenizer = load_model()
    prompts = [
        "What is the capital of France?",
        "Write a short poem about the ocean.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Classify the sentiment: 'I loved the movie.'",
        "Brainstorm three party themes.",
    ]
    for p in prompts:
        formatted = f"### Instruction:\n{p}\n\n### Response:\n"
        out = generate(model, tokenizer, formatted, temperature=0.7, max_new_tokens=60, seed=0)
        completion = out[len(formatted):].strip() if out.startswith(formatted) else out.strip()
        assert len(completion) > 0
        words = completion.split()
        if len(words) > 6:
            # no obvious 3-gram repetition loop
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            assert len(trigrams) == 0 or len(set(trigrams)) > 1
