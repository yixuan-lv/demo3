import subprocess
from pathlib import Path

def prediction_config(name="predict_qlora.yaml"):
    return Path(__file__).resolve().parents[1] / "configs" / name

def predict(config):
    from src.model import get_command
    subprocess.run(get_command(config), check=True)
