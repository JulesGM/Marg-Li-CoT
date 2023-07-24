import logging
import typing

import datasets
import torch
import transformers

import lib_base_classes

LOGGER = logging.getLogger(__name__)


class SentimentRewardFn(lib_base_classes.Reward):
    def __init__(self, ppo_trainer):
        self._pipeline = _make_sentiment_pipeline(
            pipeline_model_name="lvwerra/distilbert-imdb",
            accelerator_num_process=ppo_trainer.accelerator.num_processes,
            accelerator_device=ppo_trainer.accelerator.device,
        )
        self._reward_kwargs = {
            "function_to_apply": "none",
            "top_k": None,
        }

    def __call__(
        self,
        *,
        queries,
        responses,
        ref_answers=None,
        task_list=None,
    ) -> lib_base_classes.RewardOutput:
        if task_list is None:
            task_list = ["[positive]"] * len(queries)

        assert ref_answers is None, "ref_answers should be None with task 'sentiment'"
        reward_values = _compute_reward(
            query_no_ctrl=queries,
            reward_kwargs=self._reward_kwargs,
            generated=responses,
            reward_fn=self._pipeline,
            task_list=task_list,
        )

        optional_kwargs: dict[str, list[str]] = {}
        if task_list is not None:
            optional_kwargs["task_list"] = task_list

        return lib_base_classes.RewardOutput(
            values=reward_values,
            name="sentiment",
            logging_columns=optional_kwargs,
        )


def _make_sentiment_pipeline(
    *,
    pipeline_model_name,
    accelerator_num_process,
    accelerator_device,
):
    if accelerator_num_process == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = accelerator_device

    assert device != "cpu", "CPU is not supported for sentiment analysis"

    sentiment_pipe = transformers.pipeline(  # type: ignore
        "sentiment-analysis", pipeline_model_name, device=device
    )

    return sentiment_pipe


def _extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "POSITIVE":
                positive_logits.append(torch.tensor(element["score"]))

    return positive_logits


def _pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """

    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[neutral]":
            logit[i] = -2 * torch.abs(logit[i]) + 4
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")

    return logit


def _compute_reward(
    *,
    query_no_ctrl: list[str],
    generated: list[str],
    reward_fn,
    reward_kwargs,
    task_list,
):
    assert isinstance(query_no_ctrl[0], str), type(query_no_ctrl[0])
    assert isinstance(generated[0], str), type(generated[0])

    texts = [q + r for q, r in zip(query_no_ctrl, generated)]
    logits = _extract_pipe_output(reward_fn(texts, **reward_kwargs))

    return _pos_logit_to_reward(logits, task_list)


def prep_dataset_rl(
    any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    txt_in_len: int, 
    split: str,
):
    assert any_tokenizer.pad_token == any_tokenizer.eos_token

    dataset = typing.cast(datasets.Dataset, datasets.load_dataset("imdb", split=split))
    dataset = dataset.rename_columns({"text": "review", "label": "sentiment"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 500, batched=False)
    dataset = dataset.map(lambda x: {"review": x["review"][:1000]}, batched=False)

    dataset = dataset.map(
        lambda x: {
            "input_ids": any_tokenizer.encode(" " + x["review"], return_tensors="pt")[
                0, :txt_in_len
            ]
        },
        batched=False,
    )
    dataset = dataset.map(
        lambda x: {"query": any_tokenizer.decode(x["input_ids"])},
        batched=False,
    )
    dataset = dataset[:20480]
    dataset = datasets.Dataset.from_dict(dataset)
    dataset.set_format("pytorch")
    return dataset


def prep_dataset_sft(
        tokenizer, split, 
        maxlen_tok=None, 
        maxlen_char=None, 
        minlen_char=None,
        minlen_tok=None,
    ):
    LOGGER.warning("Not tested.")

    assert tokenizer.pad_token == tokenizer.eos_token
    dataset = typing.cast(
        datasets.Dataset, datasets.load_dataset("imdb", split=split))
    
    if minlen_char:
        dataset = dataset.filter(lambda x: len(x["text"]) > minlen_char, batched=False)
    
    if maxlen_char:
        dataset = dataset.map(lambda x: {"review": x["text"][:maxlen_char]}, batched=False)

    if maxlen_tok:
        dataset = dataset.map(
            lambda x: {"input_ids": tokenizer(x["text"])[:, :maxlen_tok]},
            batched=True,
        )

    if minlen_tok:
        
        def filter_fn(x):
            assert x.ndim == 1, x.shape
            return len(x["input_ids"]) > minlen_tok
        
        dataset = dataset.filter(
            filter_fn, 
            batched=False
        )

    dataset = dataset.map(
        lambda x: {"text": tokenizer.batch_decode(x["input_ids"])},
        batched=True,
    )

    dataset = datasets.Dataset.from_dict(dataset)  # type: ignore
    dataset.set_format("pytorch")
    
    return dataset