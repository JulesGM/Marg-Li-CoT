#!/usr/bin/env bash
# python open-instruct/open_instruct/eval/gsm/run_eval.py
import enum
import os
import pathlib
import re
import shlex
import sys
import json

import fire
import rich
import rich.panel
import rich.traceback
import subprocess

rich.traceback.install()
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

def add_to_path(new_path):
    new_path = pathlib.Path(new_path).expanduser().resolve()

    assert new_path.exists(), f"Does not exist: {new_path}"
    assert new_path.is_dir(), f"Invalid path: {new_path}"

    sys.path.append(str(new_path))

sys.path.append(str(SCRIPT_DIR / "open-instruct"))


class ExperimentGroup(str, enum.Enum):
    gsm_direct = "gsm_direct"
    gsm_cot = "gsm_cot"


class NoArgV:
    """
    A class to represent a flag that takes no arguments.
    e.g. `--use_vllm` would be represented as `use_vllm=NoArgV`

    """
    pass


def run(
    dry=False,
    experiment_group="gsm_cot", 
    use_chat_format=True,
    model_name_or_path="/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_100",
    tokenizer_name_or_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    gsm_stop_at_double_newline=True,
    data_dir=SCRIPT_DIR / "gsm8k_eval_data",
    save_dir=SCRIPT_DIR / "eval_outputs",
    max_num_examples=0,
    use_vllm=True,
):

    if not tokenizer_name_or_path:
        if pathlib.Path(model_name_or_path).is_dir():
            import transformers
            config = transformers.AutoConfig.from_pretrained(pathlib.Path(model_name_or_path) / "config.json")
            breakpoint()
            tokenizer_name_or_path = config.tokenizer_name_or_path

    #############################
    # Validate inputs
    #############################
    experiment_group = ExperimentGroup(experiment_group)
    data_dir = pathlib.Path(data_dir).expanduser().resolve()
    save_dir = pathlib.Path(save_dir).expanduser().resolve()

    assert data_dir.is_dir(), f"Invalid data_dir: {data_dir}"
    assert (data_dir / "test.jsonl").is_file(), f"Invalid script file: {(data_dir / 'test.jsonl')}"
    # assert save_dir.parent.is_dir(), f"Invalid save_dir: {save_dir}"

    #############################
    # Set up the kwargs
    #############################
    if experiment_group in {ExperimentGroup.gsm_direct, ExperimentGroup.gsm_cot}:
        # expects the GSM8K data to be in jsonl format, {"question": str, "answer": "reasoning####answer"}

        script_path = pathlib.Path("open-instruct/eval/gsm/run_eval.py")

        kwargs = dict(
            chat_formatting_function = "eval.templates.create_prompt_with_tulu_chat_format",
            data_dir                 = data_dir,
            max_num_examples         = max_num_examples,
            model_name_or_path       = model_name_or_path,
            n_shot                   = 8,
            save_dir                 = save_dir,
            tokenizer_name_or_path   = tokenizer_name_or_path,
        )
        
        if use_vllm:
            kwargs["use_vllm"] = NoArgV
        
        if experiment_group == ExperimentGroup.gsm_direct:
            kwargs["no_cot"] = NoArgV

        if gsm_stop_at_double_newline:
            kwargs["stop_at_double_newline"] = NoArgV
    
    else:
        raise ValueError(
            f"Invalid experiment group: {experiment_group}; " +
            f"must be one of {ExperimentGroup.__members__.keys()}"
        )

    if use_chat_format:
        kwargs["use_chat_format"] = NoArgV

    #############################
    # Create the command
    #############################
    script_path = pathlib.Path(script_path).expanduser().resolve()
    assert script_path.is_file(), f"Invalid script path: {script_path}"
    command = ["python", script_path]

    argument_pat = re.compile(r"^[\w_]+$")
    for k, v in sorted(kwargs.items()):
        assert argument_pat.match(k), f"Invalid argument name: {k}"

        if v is NoArgV:
            command.append(f"--{k}")
        else:
            command.extend([f"--{k}", shlex.quote(str(v))])

    rich.print(rich.panel.Panel(
        " ".join(map(str, command)).replace("--", "\n--"), 
        highlight=True, 
        title=str(model_name_or_path).split("/")[-1],
        title_align="[bold blue]left",
    ))

    #############################
    # Run the command
    #############################
    if not dry:
        save_dir.mkdir(parents=True, exist_ok=True)
        def json_handle_path(obj):
            if isinstance(obj, pathlib.Path):
                return str(obj)

            if obj is NoArgV:
                return True

            raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")

        with open(save_dir / "jules_eval_config.json", "w") as f:
            json.dump(dict(
                kwargs=kwargs,
                command=command,
            ), 
            f, 
            indent=4, 
            default=json_handle_path)
        subprocess.check_call(command)

def main(dry=False, max_num_examples=0, use_chat_format=True, use_vllm=True):
    
    save_root = SCRIPT_DIR / "eval_outputs"
    # initial_runs = [
    #     "HuggingFaceTB/SmolLM2-1.7B-Instruct", 
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_100",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_200",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_300",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_400",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_500",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_600",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_700",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_800",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_900",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_1000",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_1100",
    #     "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_1200",
    # ]

    for name in [
        # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-30_17-51-43_rlvr_8b_checkpoints/step_400"
    ]:
    # ] + list(pathlib.Path("/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-30_17-51-43_rlvr_8b_checkpoints/").glob("step_*")):

        run(
            dry                = dry,
            max_num_examples   = max_num_examples,
            model_name_or_path = name,
            save_dir           = str(save_root / str(name).replace("/", "_")),
            use_chat_format    = use_chat_format,
            use_vllm           = use_vllm,
        )

if __name__ == "__main__":
    fire.Fire(main)