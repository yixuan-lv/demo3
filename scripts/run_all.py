import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate
from src.visualization import plot_results


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_results(rows):
    result_file = ROOT / "results" / "results.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "lora_rank", "lr", "quant_bit", "precision", "recall", "f1",
        "peak_memory_gb", "train_seconds",
    ]
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_results():
    result_file = ROOT / "results" / "results.csv"
    if not result_file.exists():
        return {}
    with result_file.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


qlora = read_json(ROOT / "configs" / "qlora.json")
lora = read_json(ROOT / "configs" / "lora.json")
predict_config = read_json(ROOT / "configs" / "predict_custom.json")
output_root = Path(qlora["output_dir"]).parent
prediction_dir = ROOT / "results" / "predictions"
test_file = ROOT / "data" / "bc2gm_test.json"

experiments = [
    ("baseline", qlora, {}),
    ("rank8", qlora, {"lora_rank": 8, "lora_alpha": 16}),
    ("rank32", qlora, {"lora_rank": 32, "lora_alpha": 64}),
    ("lr1e5", qlora, {"learning_rate": 0.00001}),
    ("lr1e4", qlora, {"learning_rate": 0.0001}),
    ("lora", lora, {}),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=[item[0] for item in experiments])
    args = parser.parse_args()
    selected = set(args.experiments or [item[0] for item in experiments])
    rows = read_results()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        for name, base_config, changes in experiments:
            if name not in selected:
                continue
            train_config = dict(base_config)
            train_config.update(changes)
            train_config["output_dir"] = str(output_root / name)
            train_config["swanlab_run_name"] = f"custom-{name}"
            train_file = temp_dir / f"{name}_train.json"
            train_file.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(train_file)],
                check=True,
            )

            current_predict = dict(predict_config)
            current_predict["adapter_name_or_path"] = train_config["output_dir"]
            current_predict["quantization_bit"] = train_config.get("quantization_bit")
            predict_file = temp_dir / f"{name}_predict.json"
            predict_file.write_text(json.dumps(current_predict, indent=2), encoding="utf-8")
            prediction_file = prediction_dir / f"{name}.jsonl"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "predict.py"),
                    "--config", str(predict_file),
                    "--input", str(test_file),
                    "--output", str(prediction_file),
                    "--batch-size", "8",
                ],
                check=True,
            )

            metric = evaluate(prediction_file, test_file)
            summary = read_json(Path(train_config["output_dir"]) / "run_summary.json")
            rows[name] = {
                "name": name,
                "lora_rank": train_config["lora_rank"],
                "lr": train_config["learning_rate"],
                "quant_bit": train_config.get("quantization_bit") or "none",
                "precision": f"{metric['precision']:.4f}",
                "recall": f"{metric['recall']:.4f}",
                "f1": f"{metric['f1']:.4f}",
                "peak_memory_gb": f"{summary['peak_memory_gb']:.2f}",
                "train_seconds": f"{summary['train_seconds']:.1f}",
            }
            ordered_rows = [rows[item[0]] for item in experiments if item[0] in rows]
            write_results(ordered_rows)
            print(f"{name}: P={metric['precision']:.4f}, R={metric['recall']:.4f}, F1={metric['f1']:.4f}")

    if all(item[0] in rows for item in experiments):
        plot_results(ROOT / "results" / "results.csv", ROOT / "results")


if __name__ == "__main__":
    main()
