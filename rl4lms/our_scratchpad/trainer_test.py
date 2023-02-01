import transformers
import torch
import deepspeed


PATH = "/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/config_ds_json/config_ds_new_json.json"
HF_MODEL = "google/flan-t5-base"


def main():

    model = transformers.T5ForConditionalGeneration.from_pretrained(HF_MODEL)
    tuple_thing = deepspeed.initialize(config=PATH, model=model)
    unwrapped_model = tuple_thing[0].module

    print(f"tuple_thing: {len(tuple_thing)}")
    print(f"type(unwrapped_model):    {type(unwrapped_model)}")
    print(f"unwrapped_model.device:   {unwrapped_model.device}")



if __name__ == "__main__":
    main()