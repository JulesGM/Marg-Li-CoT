import datasets
import fire
import ipdb
import jsonlines as jl
import pathlib
import tqdm

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

def main(output_dir=SCRIPT_DIR / "gsm8k_eval_data" / "test.jsonl"):

    dataset = datasets.load_dataset("gsm8k", "main", split="test")
    with jl.open(output_dir, "w") as f:
        for example in tqdm.tqdm(dataset):
            f.write(example)

    print("Done.")

if __name__ == "__main__":
    fire.Fire(main)
