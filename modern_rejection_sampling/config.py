
import dataclasses


@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    name: str
    split: str
    valid_split: str
    question_field: str
    answer_field: str
    load_dataset_args: list

    def __post_init__(self):
        self.name = str(self.name)
        self.split = str(self.split)
        self.valid_split = str(self.valid_split)
        self.question_field = str(self.question_field)
        self.answer_field = str(self.answer_field)
        self.load_dataset_args = list(self.load_dataset_args)


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    name: str

    def __post_init__(self):
        self.name = str(self.name)


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    forward_max_length: int
    generation_max_length: int
    train_subset_mode: bool = False
    train_subset_size: int | None = None

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.num_epochs = int(self.num_epochs)
        self.forward_max_length = int(self.forward_max_length)
        self.generation_max_length = int(self.generation_max_length)
        if self.train_subset_mode:
            assert isinstance(self.train_subset_size, int), (
                f"{self.train_subset_size = } {type(self.train_subset_size).mro() = }")


@dataclasses.dataclass(kw_only=True)
class VLLMSamplingConfig:
    top_p: float = None
    temperature: float
    num_candidates: int

    def __post_init__(self):
        self.temperature = float(self.temperature)
        self.num_candidates = int(self.num_candidates)
        if self.top_p is not None:
            self.top_p = float(self.top_p)


@dataclasses.dataclass(kw_only=True)
class WandbConfig:
    project: str
    entity: str
    log_interval: int

    def __post_init__(self):
        self.project = str(self.project)
        self.entity = str(self.entity)
        self.log_interval = int(self.log_interval)


@dataclasses.dataclass(kw_only=True)
class EvaluationConfig:
    eval_subset: int = None
    eval_batch_size: int
    eval_percentage: float

    def __post_init__(self):
        if self.eval_subset is not None:
            self.eval_subset = int(self.eval_subset)
        self.eval_batch_size = int(self.eval_batch_size)
        self.eval_percentage = float(self.eval_percentage)


@dataclasses.dataclass(kw_only=True)
class AccelerateConfig:
    seed: int
    gpu_ids: list[int]

    def __post_init__(self):
        self.seed = int(self.seed)
        self.gpu_ids = [int(x) for x in self.gpu_ids]


@dataclasses.dataclass(kw_only=True)
class VLLMConfig:
    gpu_id: int

    def __post_init__(self):
        self.gpu_id = int(self.gpu_id)


@dataclasses.dataclass(kw_only=True)
class VLLMSamplingConfig:
    temperature: float
    num_candidates: int
    top_p: float = None

    def __post_init__(self):
        self.temperature = float(self.temperature)
        self.num_candidates = int(self.num_candidates)
        if self.top_p is not None:
            self.top_p = float(self.top_p)


@dataclasses.dataclass(kw_only=True)
class Config:
    accelerate: AccelerateConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    experiment_name: str
    output_dir: str
    internal_master_port: int
    master_port: int
    model: ModelConfig
    training: TrainingConfig
    vllm: VLLMConfig
    vllm_sampling: VLLMSamplingConfig
    wandb: WandbConfig
    few_shot_qty: int
    

    def __post_init__(self):
        self.accelerate = AccelerateConfig(**self.accelerate)
        self.dataset = DatasetConfig(**self.dataset)
        self.evaluation = EvaluationConfig(**self.evaluation)
        self.experiment_name = str(self.experiment_name)
        self.output_dir = str(self.output_dir)
        self.internal_master_port = int(self.internal_master_port)
        self.master_port = int(self.master_port)
        self.model = ModelConfig(**self.model)
        self.training = TrainingConfig(**self.training)
        self.vllm = VLLMConfig(**self.vllm)
        self.vllm_sampling = VLLMSamplingConfig(**self.vllm_sampling)
        self.wandb = WandbConfig(**self.wandb)
