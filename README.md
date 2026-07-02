# Small Language Model — Dolly-15k Instruction Fine-tune

## What this is
A small text-to-text instruction-following model built by **LoRA fine-tuning `distilgpt2`**
on the **Databricks Dolly-15k** dataset.

## Why fine-tune instead of train from scratch
With the time available, training a from-scratch minGPT-scale model (even at gpt-mini
size) to produce *anything* coherent requires many more tokens and iterations than fit
in a short time box. Fine-tuning a pretrained checkpoint gets a genuinely useful model
in a fraction of the compute, while LoRA keeps the trainable parameter count tiny
(fast, cheap, doesn't destroy the base model's general language ability). The trade-off
is I'm not demonstrating architecture-from-scratch skill here — that was covered in
Round 1 (minGPT walkthrough) instead.

## Why Dolly-15k
Small (13MB), single-turn (no multi-turn flattening complexity), and CC BY-SA 3.0
licensed (commercial use OK). Large enough to show a real training curve without
needing hours of data loading/tokenizing.

## Setup
```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
GPU strongly recommended (Colab free-tier T4 is enough) but not required — distilgpt2
+ LoRA on ~4k examples will run on CPU, just slower.

## Run training
```bash
python src/train.py
```
Config (dataset subset size, epochs, batch size, LR, LoRA rank/target modules) lives
at the top of `src/train.py`. Loss is logged to stdout every `logging_steps` steps.
Checkpoint is saved to `checkpoints/distilgpt2-dolly-lora/`.

## Run inference
```bash
python src/generate.py --prompt "Explain photosynthesis" --temperature 0.8 --max_new_tokens 100
```
`--temperature` and `--max_new_tokens` are adjustable CLI args (not hardcoded).

## Run evaluation
```bash
python src/evaluate.py
```
Prints validation loss, perplexity, and outputs for 8 held-out prompts spanning
Dolly's task categories (QA, creative writing, summarization, classification,
brainstorming, general knowledge, arithmetic, instruction-following).

## Run tests
```bash
pytest tests/ -v
```
Covers all 10 required scenarios (tokenizer round-trip, split leakage, overfit sanity
check, non-trivial generation, empty prompt, overlong prompt, max_new_tokens edge
cases, greedy determinism, seeded sampling, fixed-prompt quality gate). Tests that
need a trained checkpoint auto-skip if `checkpoints/` is empty, so the suite is
runnable at any point — run `train.py` first for the full pass.

## Data pipeline decisions
- Dropped examples with empty `response` (a handful in Dolly).
- Prompt format: `### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n`
  (context block omitted when absent).
- Loss is masked on the prompt tokens (label = -100), computed only on the completion —
  standard instruction-tuning practice, otherwise the model wastes capacity learning
  to predict the instruction text itself.
- Subsampled to 4,000 examples (of 15k) — documented trade-off between training-curve
  quality and time box. Increase `subset_size` in `train.py` config if you have more time.
- Train/val split: shuffle once with a fixed seed, then slice — guarantees zero overlap.

## Known limitations / error analysis
Small model, small fine-tuning set, short training run. Expect:
- Weak factual recall (dates, capitals, numeric facts often wrong or invented).
- Arithmetic will usually fail.
- Coherence degrades past 2-3 sentences.
- May not reliably follow output formatting (e.g. numbered lists for "brainstorm" prompts).

Run `src/evaluate.py` and fill in which of the 8 demo prompts actually exhibited
these before the interview — an honest, specific writeup beats a generic one.

## What I'd do differently with 10x the time/compute
- Use the full 15k Dolly examples (or add UltraChat-200k for multi-turn coverage)
  instead of a 4k subset.
- Compare LoRA against full fine-tuning to quantify the quality/compute trade-off
  (stretch goal in the assignment).
- Add an automated LLM-as-judge eval against a rubric instead of manual read-through
  of 8 prompts, to catch regressions systematically.
- Try a larger base checkpoint (e.g. Qwen2.5-0.5B) — still small enough for LoRA on a
  single GPU, likely meaningfully better factual recall and coherence.
- Add early stopping on validation loss and a learning-rate sweep instead of a fixed
  3 epochs / 2e-4.
