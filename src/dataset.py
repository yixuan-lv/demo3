import json
from pathlib import Path

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
