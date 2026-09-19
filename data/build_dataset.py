import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import convert_split

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=ROOT / "bc2gm1")
parser.add_argument("--output", type=Path, default=ROOT / "data")
args = parser.parse_args()

for split in ["train", "dev", "test"]:
    src = args.input / f"{split}.json"
    dst = args.output / f"bc2gm_{split}.json"
    count = convert_split(src, dst)
    print(f"{split}: {count} examples -> {dst}")
