import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=Path, required=True)
parser.add_argument("--gold", type=Path, default=ROOT / "data" / "bc2gm_test.json")
args = parser.parse_args()

result = evaluate(args.pred, args.gold)
print(f"Precision: {result['precision']:.4f}")
print(f"Recall:    {result['recall']:.4f}")
print(f"F1:        {result['f1']:.4f}")
