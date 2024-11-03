import dataclasses
import enum 
import pathlib
import typing
import yaml

class CodeCategory(str, enum.Enum):
    SFT = "sft"
    RL = "trl"


class Datasets(str, enum.Enum):
    ARITHMETIC = "arithmetic"
    GSM8K = "gsm8k"
    MATH = "math"


@dataclasses.dataclass
class JobConfig:
    experiment: str
    code_category: CodeCategory
    overloads: typing.Optional[dict[str, typing.Any]]
    gpu: str
    dataset: Datasets

    def __post_init__(self):
        self.code_category = CodeCategory(self.code_category)
        if self.overloads is None:
            self.overloads = {}
        assert isinstance(self.experiment, str), type(self.experiment)
        self.dataset = Datasets(self.dataset)

