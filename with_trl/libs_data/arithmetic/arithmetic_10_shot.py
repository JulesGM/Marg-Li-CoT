from pathlib import Path

import jsonlines as jl
import itertools as it

SCRIPT_DIR = Path(__file__).absolute().parent

def make_few_shots(root):
    root = Path(root)
    files = list((root / "train_scratch").glob("*.jsonl"))
    num_per = 10 // len(files)
    few_shots = []

    for file in files:
        with jl.open(file) as f:
            for sample in it.islice(f, num_per):
                few_shots.append(sample)

    assert  10 >= len(few_shots) >= 8, len(few_shots)

    return few_shots


if __name__ == "__main__":
    import json
    few_shots = make_few_shots(SCRIPT_DIR)
    print(json.dumps(few_shots, indent=4))
    print(f"Number of few shots: {len(few_shots)}")