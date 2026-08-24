from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from ablation_common.data import ExperimentVariant
from ablation_common.training import main_for_variant


if __name__ == "__main__":
    main_for_variant(ExperimentVariant.TERRAIN_GATE, Path(__file__).resolve().parent)
