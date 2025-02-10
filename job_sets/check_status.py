#!/usr/bin/env python
import fire
import subprocess
import pathlib
import inquirer.shortcuts
import jsonlines
import os
from contextlib import contextmanager
import time

import inquirer
from rich.live import Live
from rich.table import Table
import wandb

import _constants
import _ui

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def extract_log_paths(job_set_path, job_object):
    # Construct paths to logs
    if job_object.code_category == _constants.CodeCategory.SFT:
        code_cat_folder = job_object.code_category.value
    else:
        code_cat_folder = "rl"
    
    job_folder = job_set_path.parent / code_cat_folder / job_object.experiment.replace("/", "_")
    assert job_folder.exists(), f"Job folder {job_folder} not found"
    logs_out = (job_folder / "logs.out").relative_to(SCRIPT_DIR)
    logs_err = (job_folder / "logs.err").relative_to(SCRIPT_DIR)
    if not logs_out.exists():
        logs_out = "N/A"
    if not logs_err.exists():
        logs_err = "N/A"
    
    return logs_out, logs_err

def parse_path(path):
    if path is None:
        runs_dir = SCRIPT_DIR / "runs"
        run_folders = [x for x in runs_dir.iterdir() if x.is_dir()]
        sorted_by_creation_date = sorted(run_folders, key=lambda x: x.stat().st_ctime, reverse=True)
        path = pathlib.Path(inquirer.shortcuts.list_input("Select a job set", choices=[str(x) for x in sorted_by_creation_date],))

    path = pathlib.Path(path)
    assert path.exists(), f"Path {path} not found"
    if path.is_dir():
        path = path / "jobs.jsonl"

    return path

def get_job_ids():
    return set([
        int(x.split(None, 1)[0])
        for x in subprocess.check_output(["squeue", "-u", os.environ["USER"], "--noheader"], text=True).strip().split("\n")
    ])

def main(*paths):
    if not paths:
        paths = [None] # This will trigger the inquirer prompt to select a job set

    for path in paths:
        path = parse_path(path)
        active_job_ids = get_job_ids()
        api = wandb.Api()

        # Define the table with data keys, corresponding column names, and styles
        columns_main_table = {
            "code_category": ("Cat", "green"),  # Added code_category column
            "experiment": ("Experiment", "magenta"),
            "wandb_status": ("W&B Status", "grey37"),
            "job_id": ("Job ID", "cyan"),
            "slurm_status": ("SLURM Status", "blue"),
            "wandb_url": ("W&B URL", "yellow"),
        }

        columns_log_paths = {
            "code_category": ("Cat", "green"),  # Added code_category column
            "experiment": ("Experiment", "magenta"),
            "wandb_status": ("W&B Status", "grey37"),
            "logs_out": ("Logs Out", "grey42"),
            "logs_err": ("Logs Err", "grey42")
        }

        with (  _ui.LiveTable("Job Status Overview", columns_main_table, create_live_context=False) as main_table, 
                _ui.LiveTable("Log Paths"          , columns_log_paths , create_live_context=False) as log_paths_table,
                _ui.chain_live_tables(main_table, log_paths_table)
            ):
            
            with jsonlines.open(path, mode="r") as reader:
                for line in reader:
                    job_id = line["job_id"]
                    slurm_status = "[green]Active" if job_id in active_job_ids else "[cyan]Not Active"
                    job_object = _constants.JobConfig(**line["job"])

                    try:
                        wandb_run = api.run(line["wandb_id"])
                        if wandb_run.state == "finished":
                            wandb_status = "[green bold]Finished"
                        elif wandb_run.state == "running":
                            wandb_status = "[green]Running"
                        elif wandb_run.state == "failed":
                            wandb_status = "[red]Failed"
                        elif wandb_run.state == "crashed":
                            wandb_status = "[red]Crashed"
                        else:
                            wandb_status = f"State: {wandb_run.state}"
                    except wandb.CommError as err:
                        wandb_status = f"[orange4]Exception: {err}"

                    logs_out, logs_err = extract_log_paths(path, job_object)

                    # Prepare data for the row update
                    row_data = {
                        "job_id": job_id,
                        "experiment": job_object.experiment,
                        "code_category": job_object.code_category.name,  # Added code_category value
                        "wandb_status": wandb_status,
                        "slurm_status": slurm_status,
                        "wandb_url": line["wandb_url"],
                    }
                    
                    log_paths_data = {
                        "code_category": job_object.code_category.name,
                        "experiment": job_object.experiment,
                        "wandb_status": wandb_status if not "Exception:" in wandb_status else "<Exception>",
                        "logs_out": logs_out,
                        "logs_err": logs_err,
                    }

                    # Update the table with the new row
                    main_table.step(row_data)
                    log_paths_table.step(log_paths_data)

if __name__ == "__main__":
    fire.Fire(main)
