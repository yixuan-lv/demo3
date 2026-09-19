import json
import random
from pathlib import Path

import torch

def project_root():
    return Path(__file__).resolve().parent

def config_path(name="qlora.json"):
    return project_root() / "configs" / name

def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def dataset_path(name):
    return project_root() / "data" / f"{name}.json"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
