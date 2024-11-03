import dataclasses
import pathlib
import shlex

import simple_slurm

import _common
import _constants
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def _make_run_dir(*, middle, output_dir: pathlib.Path, job: _constants.JobConfig):
    assert output_dir.exists(), f"Output directory {output_dir} not found"
    first_part = output_dir / middle
    first_part.mkdir(exist_ok=True)
    run_dir = first_part / job.experiment.replace("/", "_")
    run_dir.mkdir()
    return run_dir


def launch_sft(job: _constants.JobConfig, wandb_id: str, output_root):

    # Run with sbatch
    run_script = SCRIPT_DIR.parent / "approach_sft" / "launch.py"
    assert run_script.exists(), f"Run script {run_script} not found"

    wandb_hash = wandb_id.split("/")[-1]

    command = [
        "python", 
        str(run_script), 
        f"--experiment={shlex.quote(job.experiment)}", 
        f"--wandb_id={shlex.quote(wandb_hash)}",
    ]
    
    return command, _make_run_dir(middle="sft", output_dir=output_root, job=job)

def launch_trl(job: _constants.JobConfig, wandb_id: str, output_root):
    # Run with sbatch
    run_script = SCRIPT_DIR.parent / "with_trl" / "launch.py"
    assert run_script.exists(), f"Run script {run_script} not found"

    wandb_hash = wandb_id.split("/")[-1]

    command = [
        "python", 
        str(run_script), 
        f"--experiment={shlex.quote(job.experiment)}", 
        f"--wandb_id={shlex.quote(wandb_hash)}",
    ]
    
    return command, _make_run_dir(middle="rl", output_dir=output_root, job=job)


def launch(
    *, 
    job: _constants.JobConfig, 
    output_root: str | pathlib.Path,
    launch_set_name: str,
    wandb_id: str
) -> int:
    
    output_root = pathlib.Path(output_root)
    assert output_root.exists(), f"Output root {output_root} not found"

    if job.code_category == _constants.CodeCategory.SFT:
        command, run_dir = launch_sft(job=job, wandb_id=wandb_id, output_root=output_root)
    elif job.code_category == _constants.CodeCategory.RL:
        command, run_dir = launch_trl(job=job, wandb_id=wandb_id, output_root=output_root)
    else:
        raise ValueError(f"Unknown code category {job.code_category}")

    parts = wandb_id.split("/")
    assert len(parts) == 3, parts
    wandb_url = f"https://wandb.ai/{parts[0]}/{parts[1]}/runs/{parts[2]}"

    job_name = f"{output_root.name}/{run_dir.relative_to(output_root)} || {wandb_url}"

    slurm = simple_slurm.Slurm(
        job_name=shlex.quote(job_name),
        output=run_dir / f"logs.out",
        error=run_dir / f"logs.err",
        partition="long",
        mem_per_cpu="30G",
        cpus_per_task=4,
        gres=f"gpu:{job.gpu}:1",
    )

    job_id = slurm.sbatch(shlex.join(command))

    _common.to_yaml(
        dict(
            wandb_url=wandb_url,
            job_id=job_id,
            job_name=job_name,
            wandb_id=wandb_id,
            **dataclasses.asdict(job)
        ),
        run_dir / "job_config.yaml"
    )

    return job_id