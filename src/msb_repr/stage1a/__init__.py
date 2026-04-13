"""Stage 1A package: multi-scale MSB representation learning."""

from msb_repr.stage1a.config import (
    Stage1ADatasetSpec,
    Stage1ALabelConfig,
    Stage1ASplitConfig,
    Stage1AWindowConfig,
)
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset
from msb_repr.stage1a.features import (
    STAGE1A_LONG_FEATURE_COLUMNS,
    STAGE1A_SHORT_FEATURE_COLUMNS,
)
from msb_repr.stage1a.model import Stage1AModel
from msb_repr.stage1a.profiles import STAGE1A_DATASET_PROFILES, get_stage1a_profile
from msb_repr.stage1a.trainer import Stage1ATrainer, Stage1ATrainerConfig

__all__ = [
    "STAGE1A_LONG_FEATURE_COLUMNS",
    "STAGE1A_SHORT_FEATURE_COLUMNS",
    "STAGE1A_DATASET_PROFILES",
    "Stage1ADatasetSpec",
    "Stage1ADualWindowDataset",
    "Stage1ALabelConfig",
    "Stage1AModel",
    "Stage1ASplitConfig",
    "Stage1ATrainer",
    "Stage1ATrainerConfig",
    "Stage1AWindowConfig",
    "get_stage1a_profile",
]
