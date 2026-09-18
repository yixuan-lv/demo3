import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_path
from src.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="predict_qlora.yaml")
args = parser.parse_args()

predict(config_path(args.config))
