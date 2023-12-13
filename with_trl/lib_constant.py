WANDB_NAMESPACE = "us"

# "/" Can't be in WANDB_NAMESPACE because of the apparent
# limit of one "/" in a wandb key.
assert "/" not in WANDB_NAMESPACE, WANDB_NAMESPACE