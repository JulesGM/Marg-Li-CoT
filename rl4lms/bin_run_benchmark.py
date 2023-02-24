"""

Runs the benchmark the train_text_generation benchmark script.
This exists because we hate bash.
It also allows us to add to the RL4LMs registry before running the script,
which is a big plus.

We call `import_module` to import the train_text_generation script,
because otherwise the registry is not populated. 
It also makes it easier to debug.

"""
print("Importing modules...")

import importlib.util
import inspect
import logging
import os
import typing
from pathlib import Path

import datasets
import fire
import general_utils as utils
import pretty_traceback
import rich
import rich.logging
import torch
import transformers
import yaml

import dataset_gsm8k
import metric
import policy

datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

print("Done importing modules.")
pretty_traceback.install()

SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)
LOG_LEVEL = "DEBUG"

# These are defaults that can be overriden by the command line.
# See the `main` function signature.
ROOT = Path("/home/mila/g/gagnonju/RL4LMs")

# CONFIG_PATH = ROOT / "scripts/training/task_configs/summarization/t5_ppo.yml"
# CONFIG_PATH = "/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/t5_ppo_our_policy.yml"

CONFIG_PATH = "/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/our_supervised_config.yml"
MODULE_PATH = "/home/mila/g/gagnonju/RL4LMs/scripts/training/train_text_generation.py"
RESULTS_PATH = Path("/network/scratch/g/gagnonju/rl4lms-benchmarks-summarization")

# Wandb stuff
ENTITY_NAME = "julesgm"
LOG_TO_WANDB = True
PROJECT_NAME = "rl4lms-benchmarks-summarization"
EXPERIMENT_NAME = "rl4lm_experiment"


def import_module(module_path):
    """Import module from path.

    Args:
        module_path (Path): Path to the module to import.

    Returns:
        Module: Imported module.
    """

    module_name = inspect.getmodulename(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_test_generation = import_module(MODULE_PATH)


def main(
    *,
    log_level: typing.Union[str, int] = LOG_LEVEL,
    config_path: str = CONFIG_PATH,
    entity_name: str = ENTITY_NAME,
    project_name: str = PROJECT_NAME,
    log_to_wandb: bool = LOG_TO_WANDB,
    experiment_name: str = EXPERIMENT_NAME,
    base_path_to_store_results: str = RESULTS_PATH,
):
    args = locals().copy()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    logging.basicConfig(
        level=log_level,
        format=f"[{rank + 1} / {world_size}][bold]\[%(name)s]:[/]  %(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler(markup=True)],
    )

    if rank == 0:
        LOGGER.info("[bold]Arguments:[/]")
        utils.print_dict(
            args,
            logger=LOGGER,
            log_level=logging.INFO,
        )

    assert log_to_wandb
    assert entity_name
    assert project_name

    assert isinstance(
        log_level, (str, int)
    ), type(log_level)
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)

    
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
    logging.getLogger("git.cmd").setLevel(logging.INFO)
    logging.getLogger("metrics_wordmath_datasets").setLevel(logging.INFO)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
    
    if rank == 0:
        LOGGER.info("\n[bold]Config:[/]")
        utils.print_dict(config)
    

    train_test_generation.main(
        config_path=config_path,
        entity_name=entity_name,
        log_to_wandb=log_to_wandb,
        project_name=project_name,
        experiment_name=experiment_name,
        base_path_to_store_results=base_path_to_store_results,
    )


if __name__ == "__main__":
    fire.Fire(main)
