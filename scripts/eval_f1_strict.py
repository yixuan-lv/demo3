import json, re
from collections import Counter

PRED_FILE = "/root/autodl-tmp/demo3/predict/generated_predictions.jsonl"
GOLD_FILE = "/root/demo3/data/bc2gm_test.json"
TAG_RE = re.compile(r"<gene>(.*?)</gene>", re.IGNORECASE)

def extract_ents(text):
    return [m.strip().lower() for m in TAG_RE.findall(text or "")]

preds = [json.loads(l) for l in open(PRED_FILE) if l.strip()]
golds = json.load(open(GOLD_FILE))

tp = fp = fn = 0
for p, g in zip(preds, golds):
    pred_c = Counter(extract_ents(p.get("predict", "")))
    gold_c = Counter(extract_ents(g["output"]))
    tp += sum((pred_c & gold_c).values())
    fp += sum((pred_c - gold_c).values())
    fn += sum((gold_c - pred_c).values())

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
print(f"[按出现次数] TP={tp}, FP={fp}, FN={fn}")
print(f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
