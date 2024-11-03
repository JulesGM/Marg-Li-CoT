#!/usr/bin/env python
import dataclasses
import datetime
import enum
import os
import pathlib
import uuid
import yaml

import fire
import jsonlines as jl
import rich
import rich.traceback
import simple_slurm
import subprocess
import tqdm
import wandb

import _common
import _constants
import _wandb_utils
import _launch


rich.traceback.install()
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

def check_experiment(experiments_folder_dir, experiment):
    assert experiments_folder_dir.exists(), f"Experiments folder {experiments_folder_dir} not found"
    attempted_path = experiments_folder_dir / f"{experiment}.yaml"
    assert attempted_path.exists(), (
        f"Experiment {experiment} not found in {experiments_folder_dir}: {attempted_path}")


def validate_experiment_sft(experiment):
    experiments_folder_dir = SCRIPT_DIR.parent / "approach_sft" / "config" / "experiment"
    check_experiment(experiments_folder_dir, experiment)

def validate_experiment_rl(experiment):
    experiments_folder_dir = SCRIPT_DIR.parent / "with_trl" / "config" / "experiment"
    check_experiment(experiments_folder_dir, experiment)


def load_config(config_set):
    config_set_path = SCRIPT_DIR / "config_sets" / f"{config_set}.yaml"
    assert config_set_path.exists(), f"Config set {config_set} not found"
    with open(config_set_path, "r") as f:
        config = yaml.safe_load(f)
    rich.print(config)
    valid_gpus_ = _common.valid_gpus()

    config["jobs"] = [_constants.JobConfig(**job) for job in config["jobs"]]
    for job in config["jobs"]:
        assert job.gpu in valid_gpus_, (
            f"Invalid GPU {job['gpu']} for job {job['experiment']}, {valid_gpus_ = }"
        )
        if job.code_category == _constants.CodeCategory.SFT:
            validate_experiment_sft(job.experiment)
        elif job.code_category == _constants.CodeCategory.RL:
            validate_experiment_rl(job.experiment)

    experiments_per_category = sorted(
        f"{job.code_category.value}/{job.experiment}" for job in config["jobs"]
    )
    assert _common.all_unique(experiments_per_category), (
        f"Duplicate experiments found in {config_set}: {experiments_per_category}"
    )
    
    return config


def main(config_set: str, user_id: str="julesgm"):
    assert isinstance(config_set, str), type(config_set)
    config = load_config(config_set)
    launch_set_name = f"{config_set}_{_common.file_safe_timestamp()}"
    save_dir = SCRIPT_DIR / "runs" / launch_set_name
    save_dir.mkdir()

    with jl.open(save_dir / "jobs.jsonl", mode="w") as writer:
        for job in config["jobs"]:
            project_id = _wandb_utils.make_wandb_project_id(
                code_category=job.code_category, 
                dataset_name=job.dataset,
            )
            wandb_id, run_suffix = _wandb_utils.generate_wandb_id(
                user_id=user_id, 
                project=project_id,
            )
            print(f"Launching job {job.experiment} with wandb id {wandb_id}")
            job_id = _launch.launch(
                job=job, 
                launch_set_name=launch_set_name, 
                output_root=save_dir,
                wandb_id=wandb_id, 
            )
            writer.write(dict(
                wandb_url=f"https://wandb.ai/{user_id}/{project_id}/runs/{run_suffix}",
                job_id=job_id, 
                job=dataclasses.asdict(job), 
                wandb_id=wandb_id, 
            ))


if __name__ == "__main__":
    fire.Fire(main)