import logging
import re
import typing

import datasets
import general_utils as utils
import jsonlines as jsonl
import pretty_traceback
import rich
import transformers
from beartype import beartype
from text2digits import text2digits

import libs_compute_accuracy.metrics_wordmath_datasets as metrics_wordmath_datasets
# import rl4lms.data_pools.text_generation_pool as rl4lms_pool
# import rl4lms.envs.text_generation.registry as rl4lms_registry

pretty_traceback.install()
datasets.logging.set_verbosity_error()
text2digits = text2digits.Text2Digits()
LOGGER = logging.getLogger(__name__)


def _clean_text(sample):
    return {
        k: v.replace("<<", "(").replace(">>", ")").strip() 
        for k, v in sample.items()
    }


def _split_answer_scratchpad(sample):
    scratchpad, answer = sample["answer"].split("####")
    return {
        "question": sample["question"].strip(),
        "answer": answer.strip(),
        "scratchpad": scratchpad.strip(),
    }


def _build_dataset(
    split, tokenizer, max_sum_squares, max_question_len, max_answer_len,
):
    """
    Max sum squares is the maximum of the square of the number of tokens in the
    question and the square of the number of tokens in the answer.
    This is to control the memory usage of the transformer model.
    """
    assert split in ("train", "test"), split
    # assert max_sum_squares is not None or (
    #     max_question_len is not None and max_question_len is not None
    # ), (max_sum_squares, max_question_len, max_question_len)

    dataset = datasets.load_dataset("gsm8k", "main", split=split)
    dataset = dataset.map(_clean_text).map(_split_answer_scratchpad)

    if max_sum_squares is not None:
        dataset.filter(
            lambda x: len(tokenizer(x["question"])["input_ids"]) ** 2
            + len(tokenizer(x["answer"])["input_ids"]) ** 2
            < max_sum_squares
        )

    if max_question_len is not None:
        dataset.filter(
            lambda x: len(tokenizer(x["question"])["input_ids"]) < max_question_len
        )

    if max_answer_len is not None:
        dataset.filter(
            lambda x: len(tokenizer(x["answer"])["input_ids"]) < max_answer_len
        )

    return dataset


@beartype
def _build_dataset_silver(
    data: list,
    split: str,
    tokenizer: transformers.PreTrainedTokenizerBase, 
    max_sum_squares: typing.Optional[int],
    max_question_len: typing.Optional[int],
    max_answer_len: typing.Optional[int],
):
    """
    Max sum squares is the maximum of the square of the number of tokens in the
    question and the square of the number of tokens in the answer.
    This is to control the memory usage of the transformer model.
    """
    assert split in ("train",), split
    assert isinstance(data, list), type(data)

    INPUT_KEY = "question"
    OUTPUT_KEY = "all_generated"

    if max_sum_squares is not None:
        data = filter(
            data,
            lambda x: 
            len(tokenizer(x[INPUT_KEY])["input_ids"]) ** 2
            + len(tokenizer(x[OUTPUT_KEY])["input_ids"]) ** 2
            < max_sum_squares
        )

    if max_question_len is not None:
        data = filter(
            data,
            lambda x: 
            len(tokenizer(x[INPUT_KEY])["input_ids"]) < max_question_len
        )

    if max_answer_len is not None:
        data = filter(
            data,
            lambda x: len(tokenizer(x[OUTPUT_KEY])["input_ids"]) < max_answer_len
        )

    return list(data)


def prep_tokenizer(tokenizer_or_name_or_path):
    if isinstance(tokenizer_or_name_or_path, str):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_or_name_or_path
        )
    else:
        assert isinstance(
            tokenizer_or_name_or_path, transformers.PreTrainedTokenizer
        ), type(tokenizer_or_name_or_path)
        tokenizer = tokenizer_or_name_or_path

    return tokenizer


class ZeroShotGSM8KTextGenPool:
    @classmethod
    def prepare(
        cls,
        split: str,
        tokenizer_or_name_or_path: typing.Union[str, transformers.PreTrainedTokenizer],
        max_sum_squares: int,
        max_question_len: int,
        max_answer_len: int,
    ):
        rich.print(
            f"[red bold]ZeroShotGSM8KTextGenPool.prepare: [white]split = {split}"
        )
        
        if split == "val":
            split = "test"

        tokenizer = prep_tokenizer(tokenizer_or_name_or_path)

        dataset = _build_dataset(
            split=split,
            tokenizer=tokenizer,
            max_sum_squares=max_sum_squares,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
        )

        samples = []
        for idx, item in enumerate(dataset):
            sample = dict(
                id=f"{split}_{idx}",
                meta_data={"ref_scratchpad": item["scratchpad"],},
                references=[item["answer"]],
                prompt_or_input_text=item["question"],
            )
            samples.append(sample)

        return samples


class SupervisedGSM8K:
    @classmethod
    def prepare(
        cls,
        split: str,
        tokenizer_or_name_or_path: typing.Union[str, transformers.PreTrainedTokenizer],
        max_sum_squares: typing.Optional[int],
        max_question_len: typing.Optional[int],
        max_answer_len: typing.Optional[int],
        answer_prompt: str,
    ):
        if split == "val":
            split = "test"

        tokenizer = prep_tokenizer(tokenizer_or_name_or_path)

        dataset = _build_dataset(
            split=split,
            tokenizer=tokenizer,
            max_sum_squares=max_sum_squares,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
        )

        samples = []
        for idx, item in enumerate(dataset):
            sample = dict(
                id=f"{split}_{idx}",
                meta_data={"ref_scratchpad": item["scratchpad"],},
                references=[item["scratchpad"] + answer_prompt + item["answer"]],
                prompt_or_input_text=item["question"],
            )
            samples.append(sample)

        return samples


NUM_PAT = re.compile(r"\d+(?:[\,\.]\d+)?")


class SilverSupervisedGSM8K:

    @classmethod
    def _deal_with_words(cls, text: str) -> typing.Optional[float]:
        converted = text2digits.convert(text)
        output = NUM_PAT.findall(converted)

        if not output:
            return None

        utils.debug_rank_0(LOGGER, "[bold blue]" + "#" * 80)
        utils.debug_rank_0(
            LOGGER,
            f"[bold blue]# text2digits[/]:\n"
            f" \t -> [green]source:[/]    {text}\n"
            f" \t -> [green]converted:[/] {converted}\n"
            f" \t -> [green]final:[/]     {output}",
        )
        utils.debug_rank_0(LOGGER, "[bold blue]" + "#" * 80)

        return output

    @classmethod
    def _split_fn(cls, generated_text: str) -> typing.Optional[str]:
        results = NUM_PAT.findall(generated_text)

        if results:
            # Numbers found
            output = results[-1]
        else:
            # No numbers found
            try:
                output = cls._deal_with_words(generated_text)
            except ValueError:
                output = None

            if output is not None:
                output = output[-1]
            else:
                utils.debug_rank_0(
                    LOGGER,
                    f"[red]split_fn: no numbers found. \n"
                    f"\t-> Received:[/] `{generated_text}`",
                )
                output = None

        return output
        
    @classmethod
    def prepare(
        cls,
        split: str,
        path: str,
        tokenizer_or_name_or_path: 
        typing.Union[str, transformers.PreTrainedTokenizer],
        max_sum_squares: typing.Optional[int],
        max_question_len: typing.Optional[int],
        max_answer_len: typing.Optional[int],
    ):
        
        args = locals().copy()
        
        LOGGER.info(
            "[bold]SilverSupervisedGSM8K.prepare Arguments:"
        )

        utils.print_dict(
            args, 
            None, 
            False, 
            LOGGER, 
            logging.INFO,
        )

        if split == "train":
            with jsonl.open(path) as fin:
                data = list(fin)

            tokenizer = prep_tokenizer(tokenizer_or_name_or_path)
            length_pre_filtering = len(data)

            print(f"Length of data pre filtering: {length_pre_filtering}")
            data = _build_dataset_silver(
                data=data,
                split=split,
                tokenizer=tokenizer,
                max_sum_squares=max_sum_squares,
                max_question_len=max_question_len,
                max_answer_len=max_answer_len,
            )
            assert isinstance(data, list), type(data)
            length_post_filtering = len(data)
            print(f"Length of data post filtering: {length_post_filtering}")
            print(f"Fraction kept: {length_post_filtering / length_pre_filtering:.1%}")
            
            samples = []
            for idx, item in enumerate(data):
                for generated in item["all_generated"]:
                    sample = dict(
                        id=f"{split}_{idx}",
                        meta_data={
                            "ref_answer": item["ref_answer"],
                            "generated_answer": item["generated_answer"],
                        },
                        references=[generated],
                        prompt_or_input_text=item["question"],
                    )

                    splitted = metrics_wordmath_datasets.split_fn(sample.references[0])
                    if splitted is not None and metric.convert_to_int(splitted) is not None:
                        samples.append(sample)

            print(f"Number of samples: {len(samples)}")
            print(
                f"Average number of generation per question: "
                f"{len(samples) / len(data) = :0.3}"
            )
        elif split in ["val", "test"]:
            if split == "val":
                split = "test"

            tokenizer = prep_tokenizer(tokenizer_or_name_or_path)
            dataset = _build_dataset(
                split=split,
                tokenizer=tokenizer,
                max_sum_squares=max_sum_squares,
                max_question_len=max_question_len,
                max_answer_len=max_answer_len,
            )

            samples = []
            for idx, item in enumerate(dataset):
                sample = rl4lms_pool.Sample(
                    id=f"{split}_{idx}",
                    meta_data={
                        "ref_scratchpad": item["scratchpad"],
                    },
                    references=[
                        item["scratchpad"] + " The answer is: " + item["answer"]
                    ],
                    prompt_or_input_text=item["question"],
                )
                samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance



if __name__ == "__main__":

    pool = SilverSupervisedGSM8K.prepare(
        "train",
        "gsm8k_gpt3_caden/outputs/data_goods.jsonl",
        tokenizer_or_name_or_path="google/flan-t5-xxl",
        max_sum_squares=None,
        max_question_len=None,
        max_answer_len=None,
    )
    rich.print(pool[3])


    pool = SilverSupervisedGSM8K.prepare(
        "val",
        "gsm8k_gpt3_caden/outputs/data_goods.jsonl",
        tokenizer_or_name_or_path="google/flan-t5-xxl",
        max_sum_squares=None,
        max_question_len=None,
        max_answer_len=None,
    )
    rich.print(pool[3])