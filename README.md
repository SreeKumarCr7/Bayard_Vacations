# Small Language Model — Dolly-15k Instruction Fine-tune + Retrieval

## What this is
A small text-to-text instruction-following model built by **LoRA fine-tuning `distilgpt2`**
on the **Databricks Dolly-15k** dataset, with a minimal **retrieval-augmented generation
(RAG)** layer added on top as a follow-up improvement.

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
Training and inference run on CPU. No GPU required.

## Run training
```bash
python src/train.py
```
Config (dataset size, epochs, batch size, LR, LoRA rank/target modules) lives at the
top of `src/train.py`. Loss is logged to stdout every `logging_steps` steps.
Checkpoint is saved to `checkpoints/distilgpt2-dolly-lora/`.

## Run inference
```bash
python src/generate.py --prompt "Explain photosynthesis" --temperature 0.8 --max_new_tokens 100
```
`--temperature` and `--max_new_tokens` are adjustable CLI args, not hardcoded.

With retrieval (RAG):
```bash
python src/generate.py --prompt "Where is Bengaluru?" --use_rag --max_new_tokens 60
```

## Run evaluation
```bash
python src/evaluate.py
```
Prints training loss, validation loss, validation perplexity, and generated outputs
for 8 held-out prompts spanning Dolly's task categories.

## Run tests
```bash
pytest tests/ -v
```
Covers all 10 required scenarios (tokenizer round-trip, split leakage, overfit sanity
check, non-trivial generation, empty prompt, overlong prompt, max_new_tokens edge
cases, greedy determinism, seeded sampling, fixed-prompt quality gate), plus 2
additional tests for the RAG retriever. Tests needing a trained checkpoint auto-skip
if `checkpoints/` is empty.

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
- **Dataset: full ~15,000 Dolly examples** (after filtering empty responses), 2 epochs.
  An earlier exploratory run used a smaller subset for faster iteration during setup;
  the submitted checkpoint is trained on the full dataset.

## Actual results (full Dolly-15k, 2 epochs, CPU)
- **Training loss:** 2.8146
- **Validation loss:** 2.7915
- **Validation perplexity:** 16.31
- Training loss decreased over the run (logged every 10 steps during `train.py`),
  confirming the training loop, LoRA setup, and data pipeline are correctly wired.

## Known limitations, observed directly from this checkpoint

**1. Repetition looping:**
Early generations (before adding repetition controls) would get stuck restating a
phrase near-verbatim — e.g. "Chicken and rice are ... sold in China" repeated four
times with minor rewording. Classic signature of an undertrained model that learned
surface instruction-response patterns without enough signal to know when to diverge.

**2. Factual hallucination and fabrication:**
On "What is the capital of France?" the model produced a circular non-answer instead
of "Paris." On out-of-distribution geographic prompts (e.g. "Where is Bengaluru?",
"Where is Tirupati?"), it produced fabricated or nonsensical place names entirely
disconnected from the actual answer.

**3. Topic drift with fluent, well-formed language:**
On "Explain photosynthesis," the model produced grammatically coherent output using
real scientific vocabulary but described general biology/respiration rather than
photosynthesis specifically — fluency and factual correctness are separate axes of
failure.

**4. Arithmetic failure:**
Basic multiplication questions produce nonsensical, unrelated text with no numbers.

**5. Instruction-following / formatting weakness:**
"Brainstorm three ideas..." produced one vague sentence, not a list of three.

**Mitigation applied:** `repetition_penalty=1.3` and `no_repeat_ngram_size=3` were
added to generation after repetition looping was caught by the test suite
(`test_generation_non_trivial` failed on a degenerate single-word output). This
reduces exact-phrase/token looping but does not address the underlying factual
or coherence limitations.

## RAG (retrieval-augmented generation) — implemented follow-up
Implemented a minimal RAG layer (`src/retriever.py`): a small hand-curated knowledge
base with keyword-overlap retrieval, wired into `generate.py` via a `--use_rag` flag.
When a relevant snippet is found, it's prepended as a `### Context:` block before
generation.

**Finding: retrieval works correctly, but the model uses the retrieved context
inconsistently rather than reliably.** Tested on "Where is Bengaluru?" with the
correct context (Karnataka, India) successfully retrieved and injected each time.
Across repeated runs the response quality varied widely: one run ignored the context
entirely and named an unrelated country; another loosely echoed context-adjacent
themes (tech hub, companies) but invented a fabricated state name and company names;
a third produced a garbled near-match to the correct answer ("Karnada" instead of
"Karnataka") combined with an unrelated city. This spread — from total miss, to
thematic-but-wrong, to partially-correct-but-garbled — shows the model is picking up
*some* signal from the retrieved context but cannot reliably extract and state the
specific fact within it.

**Why this happens:** most Dolly-15k training examples have an empty `context` field
(only a subset of rows populate it), so the model was not strongly trained to
condition its response on the `### Context:` section specifically. It learned general
instruction-response formatting but not context-grounding as a distinct, reliable
behavior. This is a known limitation of RAG with small or lightly fine-tuned models —
retrieval alone doesn't guarantee grounded generation; the model also needs sufficient
training or capacity to actually attend to and use what's retrieved.

**What would fix this:** fine-tune specifically on the subset of Dolly-15k examples
where the context field is populated, so the model learns to read and use Context, or
use a larger/more capable base model with stronger instruction-following ability.

**Ruled out prompt-wording as a fix:** made the context header explicitly directive
("use only this information to answer") with no retraining. This did not improve
grounding — if anything, fabricated content became more elaborate while still
garbling place names. This confirms the issue is a genuine training gap, not prompt
phrasing: the model was never taught during fine-tuning to treat context as
authoritative, and that can't be instilled through inference-time wording alone.

**Practical fix implemented — extractive mode:** added a `--extractive` flag that,
when a confident retrieval match is found, returns the retrieved fact directly
instead of asking the model to generate around it (`python src/generate.py --prompt
"..." --use_rag --extractive`). This guarantees factual accuracy for any query the
knowledge base covers, trading away the model's own phrasing/summarization — the
right trade-off given the model's demonstrated unreliability at grounded generation.

## Why more fine-tuning data alone doesn't fix factual hallucination
Instruction fine-tuning teaches response *format* and behavior, not new factual
knowledge. Whatever the model "knows" is entirely inherited from distilgpt2's original
pretraining — LoRA (147K trainable parameters, 0.18% of the model) is deliberately
low-capacity, designed to adjust behavior without overwriting the base model's
knowledge. This is why scaling the fine-tuning dataset improves instruction-format
generalization but does not resolve factual hallucination or arithmetic failures,
which are bounded by the base model's own knowledge and architecture.

## What I'd do differently with 10x the time/compute
- Fine-tune specifically on Dolly examples with populated `context` fields so the
  model learns to reliably ground responses in retrieved context (directly addresses
  the RAG finding above).
- Swap the keyword-based retriever for an embedding-based one (sentence-transformers +
  cosine similarity) for more robust retrieval over a larger knowledge base.
- Try a larger base checkpoint (e.g. Qwen2.5-0.5B or full `gpt2`) for a higher
  factual-knowledge ceiling — this would require GPU access, as CPU training time
  scales significantly with model size.
- Compare LoRA against full fine-tuning to quantify the quality/compute trade-off.
- Add an automated LLM-as-judge eval against a rubric instead of manual read-through
  of 8 prompts, to catch regressions systematically.
- Add early stopping on validation loss and a learning-rate sweep.
