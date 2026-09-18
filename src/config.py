from pathlib import Path

def project_root():
    return Path(__file__).resolve().parents[1]

def config_path(name="qlora.yaml"):
    return project_root() / "configs" / name
