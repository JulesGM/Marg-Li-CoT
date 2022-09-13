# Marg-Li-CoT
Note: Uses http://www.github.com/julesgm/general_utils/


## Steps:
First, install the `general_utils` package, here: http://www.github.com/julesgm/general_utils/

Then run the script with

```

python bin_refine.py main --strategy None --dataset_path [directory with the dataset] \
--checkpoint_path [directory to save the checkpoints] --wandb_entity [your wandb entity] \
--wandb_project [a project you own] --wandb_run_id None

```

See `Entrypoints.main` for the list of available arguments.

To resume, provide the `--wandb_run_id` argument with the run id of the run you want to resume. It should match a directory in the `checkpoint_path` directory, which are named after the run id.


## Steps
1. Generate default scratch pads with GPT3
2. Refine GPT2 with the dataset
3. Fine-tune GPT2 with marginal likelihood