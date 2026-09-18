import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate

result = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
print(f"{result['precision']:.4f},{result['recall']:.4f},{result['f1']:.4f}")
