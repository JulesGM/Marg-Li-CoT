import time
import tqdm
import wandb
import uuid

import _constants
import _common

def generate_run_suffix():
    return str(uuid.uuid4().hex[:8])


def make_wandb_project_id(
        *, 
        code_category: _constants.CodeCategory, 
        dataset_name: _constants.Datasets
    ) -> str:
    
    code_category = _constants.CodeCategory(code_category)
    if code_category == _constants.CodeCategory.SFT:
        return f"sft_arithmetic"
    elif code_category == _constants.CodeCategory.RL:
        return f"rl_{dataset_name.value}"
    else:
        raise ValueError(f"Invalid code category {code_category}")


def check_if_run_exists(run_id):
    api = wandb.Api()
    try:
        run = api.run(run_id)
        return True
    except wandb.CommError:
        return False


def make_run_id(user_id, project):
    run_suffix = generate_run_suffix()
    return f"{user_id}/{project}/{run_suffix}", run_suffix
    

def generate_wandb_id(*, user_id, project):
    while True:
        run_id, run_suffix = make_run_id(user_id, project)
        if not check_if_run_exists(run_id):
            return run_id, run_suffix
        print(f"Run {run_id} already exists, generating new run id")
        _common.wait(1)
