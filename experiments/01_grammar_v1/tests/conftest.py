import sys
from pathlib import Path

# make `from cfg.grammar import ...` resolve to this experiment's src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
