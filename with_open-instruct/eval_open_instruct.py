#!/usr/bin/env bash
# python open-instruct/open_instruct/eval/gsm/run_eval.py
import enum
import os
import pathlib
import re
import shlex
import sys

import fire
import rich


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


def main(
    experiment_group="gsm_direct", 
    use_chat_format=True,
    model_name_or_path="/home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_1200",
    tokenizer_name_or_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    gsm_stop_at_double_newline=False,
    data_dir=SCRIPT_DIR / "gsm8k_eval_data",
    save_dir=SCRIPT_DIR / "with_open-instruct/eval_outputs/",
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

    # assert data_dir.is_dir(), f"Invalid data_dir: {data_dir}"
    # assert save_dir.parent.is_dir(), f"Invalid save_dir: {save_dir}"

    #############################
    # Set up the kwargs
    #############################
    if experiment_group in {ExperimentGroup.gsm_direct, ExperimentGroup.gsm_cot}:
        # expects the GSM8K data to be in jsonl format, {"question": str, "answer": "reasoning####answer"}

        script_path = pathlib.Path("open-instruct/eval/gsm/run_eval.py")

        kwargs = dict(
            chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format",
            data_dir=data_dir,
            max_num_examples=200,
            model_name_or_path=model_name_or_path,
            n_shot=8,
            save_dir=save_dir,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_vllm=NoArgV,
        )
        
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
    for k, v in kwargs.items():
        assert argument_pat.match(k), f"Invalid argument name: {k}"

        if v is NoArgV:
            command.append(f"--{k}")
        else:
            command.extend([f"--{k}", shlex.quote(str(v))])

    rich.print(" ".join(map(str, command)))

    #############################
    # Run the command
    #############################
    os.execvp(command[0], command)

if __name__ == "__main__":
    fire.Fire(main)