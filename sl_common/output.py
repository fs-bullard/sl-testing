from datetime import datetime
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


def output_dir(test_name):
    """Create and return output/<test_name>/<timestamp>/ (gitignored)."""
    path = OUTPUT_ROOT / test_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path
