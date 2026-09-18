import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import plot_results

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, default=ROOT / "results" / "results.csv")
parser.add_argument("--output", type=Path, default=ROOT / "results")
args = parser.parse_args()

plot_results(args.csv, args.output)
print(f"Saved figures to {args.output}")
