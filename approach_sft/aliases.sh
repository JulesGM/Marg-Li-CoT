#!/usr/bin/env bash
alias log="python -c '
import pathlib
import re
import rich
logs = list(pathlib.Path.cwd().glob(\"*.out\"));
path = max(logs, key=lambda x: int(re.findall(r\"\\d+\", x.name)[-1]))
rich.print(f\"[bold blue]Path: {path}\")
rich.print(path.read_text())
'
"
