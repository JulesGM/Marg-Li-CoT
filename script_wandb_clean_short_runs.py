"""

Deletes wandb runs that are under a certain duration or that don't have a recorded duration at all.

"""

import datetime
from typing import *

from beartype import beartype
import fire
import pretty_traceback  # type: ignore
import rich
import rich.table
import wandb

import general_utils as utils

pretty_traceback.install()
NUM_MINUTES_TO_KEEP = 10
NORMALIZED_LEN      = 2
PROJECT_NAME        = "rl4lms-benchmarks-summarization"
FILL_CHAR           = " "
USERNAME            = "julesgm"


@beartype
def parse_time(seconds: int) -> tuple[int, int, int]:
    hours        =  seconds // 60  // 60
    minutes      = (seconds // 60) % 60
    norm_seconds =  seconds % 60
    return hours, minutes, norm_seconds


@beartype
def len_normalize_num_str(num: int, target_l: int, fill_char: str) -> str:
    str_num = str(num)
    if len(str_num) < target_l:
        str_num = (target_l - len(str_num)) * fill_char + str_num
    return str_num


@beartype
def main(
    *,
    project_name: str = PROJECT_NAME,
    min_minutes: int = NUM_MINUTES_TO_KEEP,
    username: str = USERNAME,
    print_defaults: bool=False,
):
    if print_defaults:
        rich.print("[bold white]Defaults:[/]")
        utils.print_dict(locals())
        exit()

    assert min_minutes < 60, (
        "Use hours if you want more than 59 minutes.")

    # Get all runs
    try:
        runs = list(wandb.Api().runs(f"{username}/{project_name}"))
    except Exception as err:
        err.args += f"{min_minutes = }"
        raise err

    table = rich.table.Table("Name", "Id",  "Runtime", "Action", "Date")
    for run in runs:
        # run.state
        maybe_timestamp_str = run.summary.get("_timestamp", None)
        name_str = f"[bold bright_cyan]{run.name}"

        if maybe_timestamp_str:
            timestamp_str = (
                f"[gray]{datetime.datetime.fromtimestamp(run.summary['_timestamp'])}"
            )
        else:
            timestamp_str = "[red]No timestamp."

        if "_runtime" in run.summary:
            hours, minutes, seconds = parse_time(round(run.summary["_runtime"]))

            seconds_str = len_normalize_num_str(seconds, NORMALIZED_LEN, FILL_CHAR)
            minutes_str = len_normalize_num_str(minutes, NORMALIZED_LEN, FILL_CHAR)
            hours_str   = len_normalize_num_str(hours,   NORMALIZED_LEN, FILL_CHAR)

            time_str = f"{hours_str}h {minutes_str}m {seconds_str}s"

            if run.state != "running" and minutes < min_minutes and hours == 0:
                table.add_row(name_str, run.id, time_str, "[red]Deleting.", timestamp_str)
                run.delete()
            else:
                table.add_row(name_str, run.id, time_str, "-", timestamp_str)

        else:
            table.add_row(name_str, "[red]No runtime.", "[red]Deleting.", timestamp_str)
            run.delete()

    rich.print(f"Minimum duration: {min_minutes} min")
    rich.print(table)


if __name__ == "__main__":
    fire.Fire(main)
