import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

INSTRUCTION = (
    "You are a biomedical named entity recognition system. "
    "Identify all GENE entities in the given sentence and wrap each one with "
    "<gene> and </gene> tags. Only output the tagged sentence, nothing else."
)


def tag_sentence(sentence, entities):
    tagged = sentence
    for entity in sorted(entities, key=lambda item: item["pos"][0], reverse=True):
        start, end = entity["pos"]
        assert sentence[start:end] == entity["name"], f"mismatch: {entity}"
        tagged = tagged[:start] + "<gene>" + tagged[start:end] + "</gene>" + tagged[end:]
    return tagged


def convert_split(source, destination):
    records = json.loads(source.read_text(encoding="utf-8"))
    converted = [
        {
            "instruction": INSTRUCTION,
            "input": record["sentence"],
            "output": tag_sentence(record["sentence"], record.get("entities", [])),
        }
        for record in records
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(converted)


def build_messages(example, with_answer=True):
    content = example["instruction"] + "\n" + example["input"]
    messages = [{"role": "user", "content": content}]
    if with_answer:
        messages.append({"role": "assistant", "content": example["output"]})
    return messages


class NERDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024, max_samples=None):
        with Path(path).open(encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        prompt_ids = self.tokenizer.apply_chat_template(
            build_messages(example, with_answer=False),
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.apply_chat_template(
            build_messages(example),
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = input_ids[:self.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        return {"input_ids": input_ids, "labels": labels}


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_length = max(len(item["input_ids"]) for item in examples)
        input_ids = []
        labels = []
        attention_mask = []
        for item in examples:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
