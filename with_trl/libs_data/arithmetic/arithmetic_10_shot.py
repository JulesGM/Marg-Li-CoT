from pathlib import Path
import collections
import jsonlines as jl
import itertools as it


SCRIPT_DIR = Path(__file__).absolute().parent

def make_few_shots(root: Path, max_digits: int, num_per: int):

    root = Path(root)
    files = list((root / "train_scratch").glob("*.jsonl"))
    files = [f for f in files if int(f.stem) <= max_digits]
    few_shots = collections.defaultdict(list)

    for file_ in files:
        with jl.open(file_) as f:
            for sample in it.islice(f, num_per):
                assert int(file_.stem) - 1 == sample["num_digits"], (
                    file_.stem - 1, sample["num_digits"])
                
                num_digits = int(file_.stem) - 1
                few_shots[num_digits].append(sample)

    for k, v in few_shots.items():
        assert len(v) == num_per

    return few_shots


if __name__ == "__main__":
    import json
    few_shots = make_few_shots(SCRIPT_DIR)
    print(json.dumps(few_shots, indent=4))
    print(f"Number of few shots: {len(few_shots)}")