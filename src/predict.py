import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.config import load_config
from src.dataset import INSTRUCTION, build_messages
from src.model import load_adapter


def predict_sentence(model, tokenizer, sentence, config):
    example = {"instruction": INSTRUCTION, "input": sentence, "output": ""}
    input_ids = tokenizer.apply_chat_template(
        build_messages(example, with_answer=False),
        tokenize=True,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.get("max_new_tokens", 512),
            do_sample=config.get("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def predict_file(model, tokenizer, input_file, output_file, config, batch_size=8):
    with Path(input_file).open(encoding="utf-8") as f:
        data = json.load(f)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(data), batch_size), desc="Predict"):
            batch = data[start:start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(item, with_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.get("cutoff_len", 1024),
            )
            device = next(model.parameters()).device
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.get("max_new_tokens", 512),
                    do_sample=config.get("do_sample", False),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            predictions = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
            for example, prediction in zip(batch, predictions):
                row = {"predict": prediction.strip(), "label": example["output"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokenizer.padding_side = old_padding_side


def load_predictor(config_file):
    config = load_config(config_file)
    model, tokenizer = load_adapter(config)
    return model, tokenizer, config
