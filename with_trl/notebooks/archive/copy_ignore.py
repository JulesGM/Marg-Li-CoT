import shutil
import fire

from pathlib import Path

def main(source, target, ignore_pat):

    source = Path(source)
    target = Path(target)

    assert source.exists(), source
    assert source.is_dir(), source
    
    assert isinstance(ignore_pat, (list, tuple,)), type(ignore_pat)

    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(*ignore_pat),
    )


if __name__ == "__main__":
    fire.Fire(main)
