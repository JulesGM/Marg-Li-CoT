"""

Deletes wandb runs that are under a certain duration or that don't have a recorded duration at all.

"""

import datetime
from typing import *

from beartype import beartype
import fire
import pretty_traceback  # type: ignore
import rich.status
import rich.table

with rich.status.Status("Importing wandb...", spinner="monkey"):
    import wandb


pretty_traceback.install()
NUM_MINUTES_TO_KEEP = 5
NORMALIZED_LEN      = 2
PROJECT_NAME        = "gsm8k"
FILL_CHAR           = " "
USERNAME            = "julesgm"


@beartype
def parse_time(seconds: int) -> tuple[int, int, int]:
    norm_seconds =  seconds % 60
    minutes      = (seconds // 60) % 60
    hours        =  seconds // 60  // 60
    return hours, minutes, norm_seconds


@beartype
def len_normalize_num_str(
    *,
    fill_char: str,
    target_l:  int,
    num:       int,
) -> str:

    str_num = str(num)
    if len(str_num) < target_l:
        str_num = (target_l - len(str_num)) * fill_char + str_num
    return str_num


def fetch_runs(
    *, 
    project_name: str,
    min_minutes:  str,
    username:     str,
):

    wandb_path = f"{username}/{project_name}"
    with rich.status.Status(
        f"Getting runs from \"{wandb_path}\"...", 
        spinner="weather",
    ):
        try:
            runs = list(wandb.Api().runs(wandb_path))
        except Exception as err:
            err.args += f"{min_minutes = }"
            raise err
    return runs


def prepare_timestamp(maybe_timestamp_str):
            
    if maybe_timestamp_str:
        timestamp = datetime.datetime.fromtimestamp(
            maybe_timestamp_str).strftime(
                "%d-%m-%Y, %H:%M:%S")
        timestamp_str = f"[gray]{timestamp}"
    else:
        timestamp_str = "[red]No timestamp."

    return timestamp_str
    

def prepare_duration(runtime):
    hours, minutes, seconds = parse_time(
        round(runtime))

    seconds_str = len_normalize_num_str(
        fill_char = FILL_CHAR,
        target_l  = NORMALIZED_LEN, 
        num       = seconds, 
    )
    minutes_str = len_normalize_num_str(
        fill_char = FILL_CHAR,
        target_l  = NORMALIZED_LEN, 
        num       = minutes, 
    )
    hours_str   = len_normalize_num_str(
        fill_char = FILL_CHAR,
        target_l  = NORMALIZED_LEN, 
        num       = hours,   
    )

    time_str = f"{hours_str}h {minutes_str}m {seconds_str}s"
    return hours, minutes, seconds, time_str


def build_table(*, rows, qty_unchanged, qty_deleted):
    table = rich.table.Table(
        "Name", 
        "Id",  
        "Runtime", 
        "Action", 
        "Date",
    )
    for row in rows:
        table.add_row(*row)
        
    table.caption = (
        f"A total of {qty_unchanged + qty_deleted} runs, "
        f"[green]{qty_unchanged} untouched[/], "
        f"[red]{qty_deleted} deleted[/]."
    )
    return table


def build_info(*, min_minutes, runs):
    rows = []
    qty_deleted = 0
    qty_unchanged = 0
    
    for run in runs:
        # run.state
        name_str = f"[bold bright_cyan]{run.name}"
        timestamp_str = prepare_timestamp(
            run.summary.get("_timestamp", None))
        
        if "_runtime" in run.summary:
            hours, minutes, seconds, time_str = prepare_duration(
                run.summary["_runtime"])

            if (
                run.state != "running"   and
                minutes   <  min_minutes and
                hours     == 0
            ):
                status_str = "[red]Deleting."
                run.delete()
                qty_deleted += 1
            else:
                status_str = "-"
                qty_unchanged += 1

        else:
            time_str = "[red]No runtime."
            status_str = "[red]Deleting."
            qty_deleted += 1
            run.delete()
            
        rows.append((
            name_str,
            run.id,
            time_str,
            status_str,
            timestamp_str,
        ))

    assert (len(rows) == qty_deleted + qty_unchanged), (
        len(rows),
        qty_deleted + qty_unchanged)
    
    return qty_unchanged, qty_deleted, rows
        

@beartype
def main(
    *,
    print_defaults: bool = False,
    project_name:   str  = PROJECT_NAME,
    min_minutes:    int  = NUM_MINUTES_TO_KEEP,
    username:       str  = USERNAME,
):

    if print_defaults:
        rich.print("[bold white]Defaults:[/]")
        for k, v in locals().items():
            rich.print(f"\t- {k}: {v}")
        exit()

    assert min_minutes < 60, (
        "Use hours if you want more than 59 minutes."
    )

    # Get all runs
    runs = fetch_runs(
        project_name = project_name,
        min_minutes  = min_minutes,
        username     = username,
    )
    
    with rich.status.Status(f"Building & Displaying Table"):
        qty_unchanged, qty_deleted, rows = build_info(
            min_minutes = min_minutes,
            runs        = runs,
        )
            
        table = build_table(
            qty_unchanged = qty_unchanged,
            qty_deleted   = qty_deleted,
            rows          = rows,
        )
        
        rich.print(f"Minimum duration: {min_minutes} min")
        rich.print(table)


if __name__ == "__main__":
    fire.Fire(main)
