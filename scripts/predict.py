import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_path
from src.predict import load_predictor, predict_file, predict_sentence

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="predict_custom.json")
parser.add_argument("--text")
parser.add_argument("--input", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

model, tokenizer, config = load_predictor(config_path(args.config))

if args.input:
    output = args.output or ROOT / "results" / "custom_predictions.jsonl"
    predict_file(model, tokenizer, args.input, output, config, args.batch_size)
    print(f"Saved predictions to {output}")
elif args.text:
    print(predict_sentence(model, tokenizer, args.text, config))
else:
    while True:
        sentence = input("Sentence (q to quit): ").strip()
        if sentence.lower() == "q":
            break
        print(predict_sentence(model, tokenizer, sentence, config))
