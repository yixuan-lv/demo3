import json
import re
from collections import Counter

TAG_RE = re.compile(r"<gene>(.*?)</gene>", re.IGNORECASE)


def extract_entities(text):
    return [match.strip().lower() for match in TAG_RE.findall(text or "")]


def score(predictions, gold, count_duplicates=False):
    tp = fp = fn = 0
    for prediction, target in zip(predictions, gold):
        predicted = extract_entities(prediction.get("predict", ""))
        expected = extract_entities(target["output"])
        if count_duplicates:
            predicted_values, expected_values = Counter(predicted), Counter(expected)
            tp += sum((predicted_values & expected_values).values())
            fp += sum((predicted_values - expected_values).values())
            fn += sum((expected_values - predicted_values).values())
        else:
            predicted_values, expected_values = set(predicted), set(expected)
            tp += len(predicted_values & expected_values)
            fp += len(predicted_values - expected_values)
            fn += len(expected_values - predicted_values)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(prediction_file, gold_file, count_duplicates=False):
    predictions = [json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = json.loads(gold_file.read_text(encoding="utf-8"))
    return score(predictions, gold, count_duplicates=count_duplicates)
