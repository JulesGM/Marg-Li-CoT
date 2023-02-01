import bin_deepspeed_experim
import fire
import json
import rich
import typing

def main(
    output_path: str="./config_ds_json/config_ds_new_json.json",
    batch_size: typing.Union[int, str]="auto",
    zero_level: int=3, 
    zero_level_3_cpu_offload: bool=True,
):
    d = bin_deepspeed_experim.make_deepspeed_config(
        batch_size=batch_size, 
        wandb_config= {
            "team": "julesgm",
            "project": "supervised_learning",
            "enabled": True,
        },
        zero_level=zero_level,
        zero_level_3_cpu_offload=zero_level_3_cpu_offload,
    )
    rich.print(d)
    with open(output_path, "w") as f:
        json.dump(d, f, indent=4)
    

if __name__ == "__main__":
    fire.Fire(main)
