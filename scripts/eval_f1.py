import json, re

PRED_FILE = "/root/autodl-tmp/demo3/predict/generated_predictions.jsonl"
GOLD_FILE = "/root/demo3/data/bc2gm_test.json"

TAG_RE = re.compile(r"<gene>(.*?)</gene>", re.IGNORECASE)

def extract_ents(text):
    return [m.strip().lower() for m in TAG_RE.findall(text or "")]

preds = []
with open(PRED_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            preds.append(json.loads(line))

with open(GOLD_FILE) as f:
    golds = json.load(f)

print(f"预测数: {len(preds)}, gold 数: {len(golds)}")

tp = fp = fn = 0
for i, (p, g) in enumerate(zip(preds, golds)):
    pred_text = p.get("predict", "")
    gold_text = g["output"]
    pred_ents = extract_ents(pred_text)
    gold_ents = extract_ents(gold_text)

    pred_set = set(pred_ents)
    gold_set = set(gold_ents)

    tp += len(pred_set & gold_set)
    fp += len(pred_set - gold_set)
    fn += len(gold_set - pred_set)

    if i < 3:
        print(f"--- 样本 {i} ---")
        print(f"  gold: {gold_ents}")
        print(f"  pred: {pred_ents}")
        print()

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"===== Entity-level 指标 =====")
print(f"TP={tp}, FP={fp}, FN={fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
