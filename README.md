# Small Language Model — Dolly-15k Instruction Fine-tune

## What this is
A small text-to-text instruction-following model built by **LoRA fine-tuning `distilgpt2`**
on the **Databricks Dolly-15k** dataset.

## Why fine-tune instead of train from scratch
Training a from-scratch minGPT-scale model to produce anything coherent requires far
more tokens and iterations than fit in the time available. Fine-tuning a pretrained
checkpoint gets a genuinely functional model in a fraction of the compute, while LoRA
keeps the trainable parameter count tiny — fast, cheap, and it doesn't overwrite the
base model's general language ability. The trade-off is I'm not demonstrating
architecture-from-scratch skill here — that was covered in Round 1 (minGPT walkthrough)
instead.

## Why Dolly-15k
Small (13MB), single-turn (no multi-turn flattening complexity), and CC BY-SA 3.0
licensed (commercial use OK).

## Setup
```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
Training and inference run on CPU. No GPU required, though one would speed things up
considerably — see the note on dataset size below.

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
`--temperature` and `--max_new_tokens` are adjustable CLI args, not hardcoded.

## Run evaluation
```bash
python src/evaluate.py
```
Prints validation loss, validation perplexity, and generated outputs for 8 held-out
prompts spanning Dolly's task categories (QA, creative writing, summarization,
classification, brainstorming, general knowledge, arithmetic, instruction-following).

## Run tests
```bash
pytest tests/ -v
```
Covers all 10 required scenarios (tokenizer round-trip, split leakage, overfit sanity
check, non-trivial generation, empty prompt, overlong prompt, max_new_tokens edge
cases, greedy determinism, seeded sampling, fixed-prompt quality gate). Tests that
need a trained checkpoint auto-skip if `checkpoints/` is empty. All 10 pass against
the trained checkpoint described below.

## LoRA configuration (as actually trained)
| Parameter        | Value    |
|------------------|----------|
| Base model       | distilgpt2 (~82M params) |
| LoRA rank (r)    | 8        |
| LoRA alpha       | 16       |
| LoRA dropout     | 0.05     |
| Target modules   | `c_attn` (GPT-2 style fused Q/K/V projection) |
| Trainable params | 147,456 (0.18% of base model) |

## Data pipeline decisions
- Dropped examples with empty `response` (a handful in Dolly).
- Prompt format: `### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n`
  (context block omitted when absent).
- Loss is masked on the prompt tokens (label = -100), computed only on the completion —
  standard instruction-tuning practice, otherwise the model wastes capacity learning
  to predict the instruction text itself.
- Train/val split: shuffle once with a fixed seed, then slice — guarantees zero overlap.
- **Dataset size used: 600 examples** (of the full ~15,000), 2 epochs. This was a
  deliberate time-box trade-off given CPU-only training — a full-dataset run was
  attempted but did not complete in the available time (estimated well beyond the
  time box on CPU). The 600-example subset was large enough to produce a real,
  observable training curve and a working checkpoint for full evaluation and testing.

## Actual results (600 examples, 2 epochs, CPU)
- **Validation loss:** 2.8757
- **Validation perplexity:** 17.74
- Training loss decreased over the run (logged every 10 steps), confirming the
  training loop, LoRA setup, and data pipeline are correctly wired.

## Known limitations, observed directly from this checkpoint

**1. Repetition looping:**
Early generations (before adding repetition controls) would get stuck restating a
phrase near-verbatim — e.g. "The plot of Romeo and Juliet in two sentences is about
Romeo and Juliet in two sentences" repeated, and "Chicken and rice are ... sold in
China" repeated four times with minor rewording. This is the classic signature of an
undertrained small model that has learned surface instruction-response patterns
without enough signal to know when to diverge or stop.

**2. Factual hallucination and fabrication:**
On "What is the capital of France?" the model produced a circular non-answer
("France is the capital of France") instead of "Paris." On a prompt about Bangalore
using informal phrasing ("ABOUT BANGALORE"), the model fell back on distilgpt2's base
pretraining priors and generated a fluent but entirely fabricated Reuters-style news
article about economic sanctions, unrelated to the actual topic. On several Indian
city/geography prompts (e.g. "Where is Bengaluru", "Where is Tirupati"), the model
produced fabricated or nonsensical place names.

**3. Topic drift with fluent, well-formed language:**
On "Explain photosynthesis," the model produced grammatically coherent output using
real scientific vocabulary (oxidation, metabolic systems, biochemical pathways) but
describing general biology/respiration rather than photosynthesis specifically —
fluency and factual correctness are separate axes of failure here.

**4. Arithmetic failure:**
"What is 15 multiplied by 7?" produced nonsensical text about geometric shapes, no
numbers at all.

**5. Instruction-following / formatting weakness:**
"Brainstorm three ideas for a birthday party theme" produced one vague circular
sentence, not a list of three distinct ideas.

**Why more data alone wouldn't fix most of this:** instruction fine-tuning teaches
response *format* and behavior, not new factual knowledge. Whatever the model "knows"
is entirely inherited from distilgpt2's original pretraining — LoRA (147K trainable
parameters, 0.18% of the model) is deliberately low-capacity, designed to adjust
behavior without overwriting the base model's knowledge. Scaling to the full 15k
examples would likely improve instruction-format generalization across more phrasing
styles, but would not be expected to resolve the factual hallucination or arithmetic
failures, since those are bounded by the base model's own knowledge and architecture,
not by fine-tuning data volume.

**Mitigation applied:** `repetition_penalty=1.3` and `no_repeat_ngram_size=3` were
added to generation after the repetition looping was caught by the test suite
(`test_generation_non_trivial` failed on a degenerate single-word output). This
reduces exact-phrase/token looping but does not address the underlying factual or
coherence limitations.

## What I'd do differently with 10x the time/compute
- Complete a full-15k-example training run (likely requiring GPU access — Colab or
  similar — given how long it took on CPU even at 600 examples).
- Implement a minimal RAG (retrieval-augmented generation) layer to ground responses
  in retrieved facts, which is the most direct fix for the factual hallucination
  observed above — fine-tuning changes behavior, not knowledge, so this is the right
  lever rather than more fine-tuning data.
- Try a larger base checkpoint (e.g. Qwen2.5-0.5B) for a higher factual-knowledge
  ceiling to fine-tune on top of.
- Compare LoRA against full fine-tuning to quantify the quality/compute trade-off.
- Add an automated LLM-as-judge eval against a rubric instead of manual read-through
  of 8 prompts, to catch regressions systematically.
- Add early stopping on validation loss and a learning-rate sweep.
