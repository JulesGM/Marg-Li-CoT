import dataclasses
import typing
import hydra
import hydra.core.config_store
from typing import Any, Optional, Union
import omegaconf


def get_hydra_experiment_choice():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return omegaconf.OmegaConf.to_container(hydra_cfg.runtime.choices)["experiment"]


@dataclasses.dataclass
class AccMaintainHydra:
    class_name: str
    limit_to_respect: float


@dataclasses.dataclass
class PPOConfigHydra:
    kl_penalty: str
    ratio_threshold: int
    adap_kl_ctrl: bool
    init_kl_coef: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    ppo_epochs: int

    
@dataclasses.dataclass
class PeftConfigHydra:
    bias: str
    inference_mode: bool
    lora_dropout: float
    lora_alpha: int
    r: int
    task_type: Optional[str]


@dataclasses.dataclass
class ModelConfigHydra:
    model_name: str
    we_pretrained_it: bool
    adapter_path: Optional[str]


@dataclasses.dataclass
class BaseConfigHydra:
    eval_every: int
    eval_subset_size: Optional[int]
    task_name: str
    dataset_name: str

    just_start_metrics: bool
    mini_batch_size: int
    train_generation_batch_size: int
    eval_batch_size: int
    eval_generation_batch_size: int

    train_generation_kwargs: dict
    eval_generation_kwargs: dict
    model: ModelConfigHydra
    peft_config: PeftConfigHydra
    ppo_config: PPOConfigHydra
    acc_maintain: AccMaintainHydra
    curriculum_schedule: list
    learning_rate: float

    tok_max_query_length: typing.Optional[int]
    tok_max_answer_length: typing.Optional[int]
    tok_max_total_length: typing.Optional[int]
    
    answer_only_max_length: int

    use_few_shots: bool
    few_shot_qty: int
    wandb_project: str
    use_peft: bool
    max_epochs: int
    no_training: bool

    use_curriculum: bool
    inspect_indices: bool
    answer_only: bool
    answer_only_path: typing.Optional[str]

    disable_adapter_value_pretrain: bool
    value_pretrain_epochs: typing.Optional[int]
    value_pretrain_steps: typing.Optional[int]

    precision: str
    float32_precision_generation: str
    float32_precision_forward_backward: str
    peft_do_all_lin_layers: bool
    reward_type: str
    start_eval: bool

    sub_generation_batch_size: int
    arithmetic_dataset_root_folder_dir: typing.Optional[str]
    
    ppo_batch_size: int

def register_configs() -> None:
    cs = hydra.core.config_store.ConfigStore.instance()

    cs.store(
        name="config",
        node=BaseConfigHydra,
    )
    
    cs.store(
        group="acc_maintain",
        name="acc_maintain",
        node=AccMaintainHydra,
    )
    
    cs.store(
        group="peft_config",
        name="peft_config",
        node=PeftConfigHydra,
    )
    
    cs.store(
        group="ppo_config",
        name="ppo_config",
        node=PPOConfigHydra,
    )

    cs.store(
        group="model",
        name="model_config",
        node=ModelConfigHydra,
    )