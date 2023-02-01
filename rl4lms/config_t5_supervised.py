from pathlib import Path

import fire
import yaml

HF_MODEL_NAME = "google/flan-t5-base"

PAYLOAD = {
    "tokenizer": {
        "model_name": HF_MODEL_NAME,
        "padding_side": "left",
        "pad_token_as_eos_token": False,
    },
    "datapool": {
        "id": "supervised_gsm8k_text_gen_pool",
        "args": {
            "tokenizer_or_name_or_path": HF_MODEL_NAME,
            "max_sum_squares": 41957,
            "max_question_len": 112,
            "max_answer_len": 192,
            "answer_prompt": "The anwser is: ",
        },
    },
    "alg": {
        "id": "supervised",
        "training_args": {
            "per_device_train_batch_size": 16,
            "logging_steps": 5000,
            "num_train_epochs": 2,
            "weight_decay": 0.1,
            "lr_scheduler_type": "cosine",
            "learning_rate": 0.0001,
            "save_total_limit": 1,
        },
        "model_type": "seq2seq",
        "model_name": HF_MODEL_NAME,
        "generation_kwargs": {
            "do_sample": True,
            "top_k": 0,
            "temperature": 1,
            "min_length": 50,
            "max_new_tokens": 192,
            "post_processing_fn": None,
        },
    },
    "train_evaluation": {
        "eval_batch_size": 100,
        "metrics": [
            {"id": "word_math_int_scratchpad_answer_accuracy", "args": {}},
            {"id": "meteor", "args": {}},
            {"id": "rouge"},
            {"id": "bleu", "args": {}},
            {"id": "bert_score", "args": {"language": "en"}},
            {"id": "diversity", "args": {}},
            # {"id": "perplexity", "args": {
            #     "model_type": "seq2seq",
            #     "tokenizer_id": HF_MODEL_NAME,
            # }
            # },
        ],
    },
}


def main(output_path=Path("our_supervised_config.yml")):
    args = locals().copy()
    import rich

    rich.print("[bold]Arguments[/bold]:")
    rich.print(args)
    print()

    rich.print("[bold]Payload[/bold]:")
    rich.print(PAYLOAD)

    with open(output_path, "w") as f:
        config = yaml.dump(PAYLOAD)
        f.write(config)


if __name__ == "__main__":
    fire.Fire(main)
