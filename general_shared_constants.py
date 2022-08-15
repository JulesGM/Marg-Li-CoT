import enum


class Steps(str, enum.Enum):
    REFINING = "refining"
    FINE_TUNING = "fine_tuning"


class RefiningModes(str, enum.Enum):
    DISTILLATION = "distillation"
    TEXT = "text"


class CVSets(str, enum.Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"