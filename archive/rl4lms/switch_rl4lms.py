#!/usr/bin/env python
import pkgutil
import shlex
import subprocess
from pathlib import Path

import fire
import more_itertools
import rich

paths = {
    Path("/home/mila/g/gagnonju/unmod_rl4lms_acc/RL4LMs/"),
    Path("/home/mila/g/gagnonju/AccelerateRL4LMS/"),
}

def cmd(cmd):
    joined = shlex.join(cmd)
    rich.print(f"[bold blue]Running command: [white]{joined}")
    rich.print(subprocess.check_output(cmd).decode().strip() + "\n")


def main(show=False):
    current = Path(pkgutil.get_loader('rl4lms').get_filename()).parent.parent
    assert current in paths, (current, *paths)
    
    other = more_itertools.one([x for x in paths if x != current])
    rich.print(f"[bold blue]current: [white]{current}")
    rich.print(f"[bold blue]other:   [white]{other}")
    if show:
        exit()


    cmd_uninst = ["conda", "develop", str(current), "--uninstall"]
    cmd_inst = ["conda", "develop", str(other)]
    
    print()
    cmd(cmd_uninst)
    cmd(cmd_inst)
    rich.print(f"[bold blue]Done! Active is now: [white]{other}")


if __name__ == "__main__":
    fire.Fire(main)
