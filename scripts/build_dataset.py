import json
import os

SRC_DIR = "/root/demo3/bc2gm1"
DST_DIR = "/root/demo3/data"
os.makedirs(DST_DIR, exist_ok=True)

INSTRUCTION = (
    "You are a biomedical named entity recognition system. "
    "Identify all GENE entities in the given sentence and wrap each one with <gene> and </gene> tags. "
    "Only output the tagged sentence, nothing else."
)

def tag_sentence(sentence, entities):
    """把实体按起始位置从后往前替换，避免位置偏移"""
    tagged = sentence
    # 按 start 从大到小排序，从后往前替换
    ents = sorted(entities, key=lambda e: e["pos"][0], reverse=True)
    for e in ents:
        start, end = e["pos"]
        assert sentence[start:end] == e["name"], f"mismatch: {e}"
        tagged = tagged[:start] + "<gene>" + tagged[start:end] + "</gene>" + tagged[end:]
    return tagged

def convert(split):
    src = os.path.join(SRC_DIR, f"{split}.json")
    dst = os.path.join(DST_DIR, f"bc2gm_{split}.json")
    with open(src) as f:
        data = json.load(f)
    out = []
    for d in data:
        sent = d["sentence"]
        ents = d.get("entities", [])
        out.append({
            "instruction": INSTRUCTION,
            "input": sent,
            "output": tag_sentence(sent, ents),
        })
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{split}: {len(out)} examples -> {dst}")

for s in ["train", "dev", "test"]:
    convert(s)