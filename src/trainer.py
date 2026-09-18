import subprocess
from pathlib import Path

def training_config(name="qlora.yaml"):
    return Path(__file__).resolve().parents[1] / "configs" / name

def train(config):
    from src.model import get_command
    subprocess.run(get_command(config), check=True)
