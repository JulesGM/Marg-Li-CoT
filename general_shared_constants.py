import enum


class RefiningModes(str, enum.Enum):
    DISTILLATION = "distillation"
    TEXT = "text"


class PipelineModes(str, enum.Enum):
    MARGINAL_LIKELIHOOD_TRAINING = "marginal_likelihood_training"
    MLE_TRAINING = "mle_training"
    VALIDATION = "validation"
    TEST = "test"


class CVSets(str, enum.Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


PIPELINES_MODES_TO_CV_SETS = {
    PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: CVSets.TRAINING,
    PipelineModes.MLE_TRAINING: CVSets.TRAINING,
    PipelineModes.VALIDATION: CVSets.VALIDATION,
    PipelineModes.TEST: CVSets.TEST,
}