#!/usr/bin/env bash

find . -maxdepth 1 -iname "slurm*out" | pawk -B '
squeue = subprocess.check_output("squeue -u gagnonju --format %i", shell=True, text=True)
no_header_split = squeue.strip().split("\n")[1:]
job_ids = set(no_header_split)
' \
'
id_ = re.findall("\d+", f[0])[-1]
if id_ not in job_ids:
    print(id_)
    
'