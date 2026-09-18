import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_path
from src.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="qlora.yaml")
args = parser.parse_args()

train(config_path(args.config))
