import collections
import enum
import transformers


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


class DataModes(str, enum.Enum):
    JSONL = "jsonl"
    HDF5_PRETOK = "hdf5_pretok"


class ModelModes(str, enum.Enum):
    """
    Whether to start with a pretrained or random model.
    Not sure why we don't just use a bool. Maybe there could 
    be more options in the future.
    """
    PRETRAINED = "pretrained"
    RANDOM = "random"


class TokenizerModes(str, enum.Enum):
    """
    
    Pretrained mode uses the Huggingface pretrained tokenizer.
    Arithmetic mode uses our custom tokenizer in data_tokenizer.py. 

    """
    PRETRAINED = "pretrained"
    ARITHMETIC = "arithmetic"


class TokenizerPaddingModes(str, enum.Enum):
    """
    
    GPT2 is not pre-trained with a special token for padding.
    - USE_EOS: use the end-of-sentence token as padding
    - NEW_TOKEN: use a new token as padding

    Using a separate token for padding is not necessary in theory. Indeed,
    the model should never back-propagate to them, they should always be masked.

    The only reason to use a separate token is to test masking more directly.
    Indeed, if we use a special padding token, we can just check that
    
    ```
    (attention_mask == (input_ids != padding_token_id)).all()
    ```
    
    We can't do this if we use the end-of-sentence token as padding, 
    because of the end-of-sentence token's regular role.

    """
    USE_EOS = "use_eos"
    NEW_TOKEN = "new_token"


class SchedulerTypes(str, enum.Enum):
    """
    The type of scheduler to use.
    """
    CONSTANT = "constant"
    LINEAR_WARMUP_CONSTANT = "linear_warmup_constant"
    LINEAR_WARMUP_LINEAR = "linear_warmup_linear"


class LMMaskingMode(str, enum.Enum):
    """
    The type of masking to use for language modeling.
    """
    PLAIN_AUTOREGRESSIVE = "PLAIN_AUTOREGRESSIVE"
    MASK_INPUT = "mask_input"