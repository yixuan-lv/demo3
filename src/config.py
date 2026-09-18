import json
from pathlib import Path

def project_root():
    return Path(__file__).resolve().parents[1]

def config_path(name="qlora.json"):
    return project_root() / "configs" / name

def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def dataset_path(name):
    return project_root() / "data" / f"{name}.json"
