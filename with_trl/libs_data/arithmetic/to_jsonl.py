import jsonlines
import fire
from pathlib import Path
import rich
import contextlib
import rich.rule

SCRIPT_DIR = Path(__file__).absolute().parent


def main(in_path=SCRIPT_DIR, out_root=SCRIPT_DIR, dry=False):
    in_path = Path(in_path)
    out_root = Path(out_root)
    assert in_path.exists(), in_path
    assert out_root.exists(), out_root
    out_path = out_root / "outputs"
    out_path.mkdir(exist_ok=True)
    
    dirs = [
        x for x in in_path.iterdir() 
        if x.is_dir() and ("train" in x.name or "val" in x.name)
    ]
    
    rich.print(dirs)
    for dir_ in dirs:
        rich.print(rich.rule.Rule())
        files = sorted(dir_.glob("*.txt"), key=lambda x: int(x.name.split(".")[0]))

        rich.print("Dir is:")
        rich.print(f"\t - {dir_}")
        rich.print("Files are:")
        for file_ in files:
            rich.print(f"\t - {file_}")
        
        target_dir = out_path / dir_.name
        if not dry:
            target_dir.mkdir(exist_ok=False)
        
        for file_ in files:
            num_digits = int(file_.name.split(".")[0])
            target = target_dir / f"{dir_.name}.{num_digits}.jsonl"
            text = file_.read_text().strip()
            entries = text.split("<|endoftext|>")
            assert entries[-1].strip() == ""
            entries = entries[:-1]

            rich.print("Doing:")
            rich.print(f"\t - Source:      {file_}")
            rich.print(f"\t - Target:      {target}")
            rich.print(f"\t - Num entries: {len(entries)}")
            
            with (jsonlines.open(target, "w") if not dry else contextlib.nullcontext()) as fout:
                
                for entry in entries:        
                    lines = entry.strip().split("\n")
                    entry_dict = dict(
                        input =      lines[1],
                        answer =     lines[-1],
                        num_digits = num_digits,
                    )
                    
                    if "<scratch>" in entry:
                    
                        assert "</scratch>" in entry
                        assert lines[3]  == "<scratch>", lines[3]
                        assert lines[-2] == "</scratch>", lines[-2]
                        
                        entry_dict["scratchpad"] = "\n".join(lines[3:-1])
                        
                    if not dry:
                        fout.write(entry_dict)
                    else:
                        print(entry_dict)


if __name__ == "__main__":
    fire.Fire(main)
