import collections
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
    PipelineModes.MARGINAL_LIKELIHOOD_TRAINING.value: CVSets.TRAINING,
    PipelineModes.MLE_TRAINING: CVSets.TRAINING,
    PipelineModes.VALIDATION: CVSets.VALIDATION,
    PipelineModes.TEST: CVSets.TEST,
}

CV_SETS_TO_PILELINES_MODES = collections.defaultdict(set)
for k, v in PIPELINES_MODES_TO_CV_SETS.items():
    CV_SETS_TO_PILELINES_MODES[v].add(k)