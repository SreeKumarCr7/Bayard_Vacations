"""
Inference script: load base model + LoRA adapter, generate from a prompt.

Usage:
    python src/generate.py --prompt "Explain photosynthesis" --temperature 0.8 --max_new_tokens 100
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "distilgpt2"
DEFAULT_CHECKPOINT = "checkpoints/distilgpt2-dolly-lora"


def load_model(checkpoint_dir: str = DEFAULT_CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, temperature: float = 0.8,
             max_new_tokens: int = 100, do_sample: bool = True, seed: int | None = None) -> str:
    """
    Handles edge cases:
    - empty prompt -> falls back to BOS/EOS token so generation doesn't crash
    - prompt longer than context window -> truncated from the left (keep most recent context)
    - max_new_tokens=0 -> returns prompt echo with no new tokens
    - very large max_new_tokens -> capped at a sane ceiling
    """
    if seed is not None:
        torch.manual_seed(seed)

    if prompt == "":
        prompt = tokenizer.eos_token

    max_new_tokens = max(0, min(max_new_tokens, 512))  # enforced cap

    model_max_len = getattr(model.config, "n_positions", 1024)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max(1, model_max_len - max_new_tokens),
    )

    if max_new_tokens == 0:
        return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        repetition_penalty=1.3,      # penalizes tokens already generated -> discourages loops
        no_repeat_ngram_size=3,      # hard-blocks repeating any 3-gram verbatim
    )
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-4)
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    formatted_prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    result = generate(
        model, tokenizer, formatted_prompt,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    print(result)