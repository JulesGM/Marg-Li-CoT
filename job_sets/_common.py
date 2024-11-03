import dataclasses
import datetime
import pathlib
import time
import yaml
import subprocess

import tqdm


def to_yaml(dict_obj, path):
    path = pathlib.Path(path)
    assert path.parent.exists(), f"Parent directory {path.parent} not found"
    assert not path.exists(), f"File {path} already exists"

    with open(path, "w") as f:
        yaml.dump(dict_obj, f)


def wait(duration):
    assert isinstance(duration, int), type(duration)
    for _ in tqdm.trange(duration, desc="Waiting"):
        time.sleep(1)


def file_safe_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def all_unique(iterable):
    seen = set()
    for val in iterable:
        if val in seen:
            return False
        seen.add(val)
    return True


def valid_gpus():
    savail = subprocess.check_output(
        ["savail"], 
        text=True,
    ).strip().split("\n")[2:]
    gpus = [x.split(None, 1)[0] for x in savail]
    assert all_unique(gpus)
    return set(gpus)