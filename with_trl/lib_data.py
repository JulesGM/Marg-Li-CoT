""" Datasets parsing and loading. """

import collections
import collections.abc
import enum
import logging
import math
import os
import re
import time
import typing
import xml
import xml.etree
from pathlib import Path

import datasets
import more_itertools
import numpy as np
import rich
import rich.box
import rich.table
import torch
import torch.utils
import torch.utils.data
import transformers
import wget
from text2digits import text2digits

import lib_sentiment_specific
import lib_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class MovingAverage:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._window = np.zeros(window_size)
        self._pointer = 0
        self._size = 0

    @property
    def window_size(self):
        return self._window_size

    @property
    def size(self):
        return self._size

    def update(self, value: float):
        self._window[self._pointer] = value
        self._pointer = (self._pointer + 1) % self._window_size
        self._size = min(self._size + 1, self._window_size)
        return self.get()

    def get(self) -> tuple[float, tuple[float, int]]:
        if self._size == 0:
            raise ValueError("No data in the moving average window. " "self._size == 0")

        window_sum = typing.cast(float, self._window.sum())
        return window_sum / self._size, (window_sum, self._size)


class ConvToNum:
    def __init__(self, failure_rate_moving_average_window_size: int = 10000):
        self._failure_rate = MovingAverage(failure_rate_moving_average_window_size)
        self._change_rate = MovingAverage(failure_rate_moving_average_window_size)
        self._no_answer_rate = MovingAverage(failure_rate_moving_average_window_size)
        self._converter_inst = text2digits.Text2Digits(convert_ordinals=False)
        self._answer_pat = re.compile(r"(\d+(,\d+)*)(\.\d+)?")

    @classmethod
    def _log_diffs(cls, *, level, initial, text, converted, final, change_stats):
        text_diff = "[on black] ".join(
            [
                (f"[on black]{pre_conv}" if pre_conv == conv else f"[on red]{pre_conv}")
                for pre_conv, conv in zip(text.split(), converted.split())
            ]
        )
        converted_diff = "[on black] ".join(
            [
                (f"[on black]{conv}" if pre_conv == conv else f"[on red]{conv}")
                for pre_conv, conv in zip(text.split(), converted.split())
            ]
        )
        final_diff = "".join(
            [
                (f"[on black]{final_}" if final_ == conv else f"[on red]{final_}")
                for final_, conv in zip(
                    final,
                    converted,
                )
            ]
        )

        if change_stats:
            change_str = (
                f"{change_stats[0]:0.1%}, {change_stats[1][0]}/{change_stats[1][1]}"
            )
        else:
            change_str = ""

        LOGGER.log(
            level,
            f"\n[green bold]Converted words to numbers.\n"
            + (f"change_stats: {change_str}\n" if change_stats else "")
            + f"{initial}\n"
            + f"[green]-> with words ->[white]\n"
            + f"{text_diff}\n"
            + f"[green]-> with numbers ->[white]\n"
            + f"{converted_diff}\n"
            + (
                ""
                if converted != final
                else f"[green]-> final ->[white]\n{final_diff}\n"
            )
            + "#" * 80
            + "\n",
        )

    @classmethod
    def _log_answers(
        cls,
        *,
        level,
        initial_answer,
        intermediate_answer,
        new_answer,
        final_answer,
    ):
        LOGGER.log(
            level,
            "\n"
            + "#" * 80
            + "\n"
            + f'[bold blue]initial_answer:[white]       "{initial_answer}"\n'
            + f'[bold blue]intermediate_answer:[white]  "{intermediate_answer}"\n'
            + f'[bold blue]new_answer:[white]           "{new_answer}"\n'
            + f'[bold blue]final_answer:[white]         "{final_answer}"\n',
        )

    def _conv_words_to_numbers(self, text: str) -> str:
        #######################################################################
        # First we do a variety of fixes to the text to make the inputs
        # compatible with the text2digits library.
        #######################################################################

        initial = text

        # A)
        # Remove the commas in the numbers. They break the text2digits library.
        # 1,000 -> 1000

        text_ = re.sub(r"(\d),(\d)", r"\1\2", text)
        if text_ != text:
            LOGGER.debug(
                "[bold blue]A) Removed commas in numbers:\n"
                f"[white]{text}\n"
                f"[green]{text_}\n"
            )
            text = text_

        # B)
        # Add a zero to the start of decimal numbers that start with a dot.
        # It breaks the text2digits library.
        # .5 -> 0.5

        text_ = re.sub(r"(?P<nan>\D)\.(?P<num>\d)", r"\g<nan> 0.\g<num>", text)
        if text_ != text:
            LOGGER.debug(
                "[bold blue]B) Added one or more zeros:\n"
                f"[white]{text}\n"
                f"[green]{text_}\n"
            )
            text = text_

        # C)
        # If a number is followed by a dot and a non number character or nothing,
        # text2digits breaks. We add a space between the number and the dot.
        # 1. -> 1 .

        text_ = re.sub(r"(\d)\.(?P<non_num>\D|$)", r"\1 .\g<non_num>", text)
        if text_ != text:
            LOGGER.debug(
                "[bold blue]C) Added a space after the dot:\n"
                f"[white]{text}\n"
                f"[green]{text_}\n"
            )
            text = text_

        # D)
        # A dollat sign followed by a decimal number breaks the text2digits library.
        # $1.5 -> $ 1.5
        text_ = re.sub(r"\$(\d+)\.(?P<non_num>\d|$)", r"$ \1.\g<non_num>", text)
        if text_ != text:
            LOGGER.debug(
                "[bold blue]D) Added a space after the dot:\n"
                f"[white]{text}\n"
                f"[green]{text_}\n"
            )
            text = text_

        # E)
        # Add space between the last number and a percentage sign.
        text_ = re.sub(r"(\d)%", r"\1 %", text)
        if text_ != text:
            LOGGER.debug(
                "[bold blue]E) Added a space before the percentage sign:\n"
                f"[white]{text}\n"
                f"[green]{text_}\n"
            )
            text = text_

        #######################################################################
        # Now we attempt to convert the words to numbers.
        #######################################################################
        try:
            converted = self._converter_inst.convert(text)
        except (ValueError, ArithmeticError):
            self._failure_rate.update(1)
            ratio, (sum_, size) = self._failure_rate.get()
            LOGGER.debug(
                f"\n[red bold] Failed to convert words to numbers. "
                f"(Failure rate: {ratio:0.2%}, {int(sum_)}/{size}):\n[white]"
                f"{text}"
            )
            return initial

        # Undo the changes we made to the text to make it compatible with the
        # text2digits library.

        final = re.sub(r"(\d) \.", r"\1.", converted)

        #######################################################################
        # Now we decide if it's worth returning the converted text.
        #######################################################################
        self._failure_rate.update(0)

        initial_answer = self.extract_answer(initial)
        initial_answer = initial_answer.group(0) if initial_answer else None
        intermediate_answer = self.extract_answer(text)
        intermediate_answer = (
            intermediate_answer.group(0) if intermediate_answer else None
        )
        new_answer = self.extract_answer(converted)
        new_answer = new_answer.group(0) if new_answer else None
        final_answer = self.extract_answer(final)
        final_answer = final_answer.group(0) if final_answer else None

        self._log_answers(
            level=logging.DEBUG,
            initial_answer=initial_answer,
            intermediate_answer=intermediate_answer,
            new_answer=new_answer,
            final_answer=final_answer,
        )

        self._log_diffs(
            level=logging.DEBUG,
            initial=initial,
            text=text,
            converted=converted,
            final=final,
            change_stats=None,
        )

        # We only return the text if it has changed.
        if intermediate_answer != new_answer:
            self._change_rate.update(1)
            self._log_diffs(
                level=logging.DEBUG,
                initial=initial,
                text=text,
                converted=converted,
                final=final,
                change_stats=self._change_rate.get(),
            )
            self._change_rate.update(0)
            if final_answer is None or not final_answer:
                self._no_answer_rate.update(1)
                ratio, (sum_, size) = self._no_answer_rate.get()
                LOGGER.info(
                    f"[red bold] No answer found in final text. "
                    f"{ratio:0.1%} {sum_}/{size}"
                )
            else:
                self._no_answer_rate.update(0)
            return final
        else:
            self._change_rate.update(0)
            if initial_answer is None or not initial_answer:
                self._no_answer_rate.update(1)
            else:
                self._no_answer_rate.update(0)
            return initial

    def extract_number_matches(self, text):
        """We return the match objects because they contain positional info."""

        text = self._conv_words_to_numbers(text)
        output = re.finditer(self._answer_pat, text)

        if output is None:
            # LOGGER.debug(
            #   f"[dark_orange bold on white]EXTRACT NUMBERS FAILED: `{text}`"
            # )
            pass

        return output

    def extract_number_strings(self, text):
        """Helper function to just return strings instead of matches."""
        output = self.extract_number_matches(text)
        if output is None:
            return

        return [m.group(0) for m in output]

    def extract_numbers(self, text):
        """Helper function to just return numbers instead of matches."""
        output = self.extract_number_strings(text)
        if output is None:
            return
        return [float(m) for m in output]


class ASDiv(torch.utils.data.Dataset):
    def __init__(self, *, tokenizer, cache_path, quiet=False, url=None):
        self._ds = self._populate_ds(
            cache_path=cache_path,
            quiet=quiet,
            url=url,
        )
        self._tokenizer = tokenizer

        # Check that the keys are correct.
        for inner_item in self._ds:
            new_keys = {"question", "answer"}
            assert not any(k in inner_item for k in new_keys), new_keys - (
                new_keys & set(inner_item)
            )

        super().__init__()

    @classmethod
    def _populate_ds(
        cls,
        cache_path,
        url=None,
        quiet=False,
    ):
        if url is None:
            url = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"

        cache_path = Path(cache_path)
        url = url
        data = {}

        if not cache_path.exists():
            if not quiet:
                print("Downloading dataset...")
            wget.download(url, out=str(cache_path), bar=None)  # type: ignore
            if not quiet:
                print("Download complete.")

        if not quiet:
            print("Parsing dataset...")

        with cache_path.open() as fp:
            root = xml.etree.ElementTree.parse(fp).getroot()[0]  # type: ignore
            data = [
                {element.tag: element.text for element in x} | dict(x.items())
                for x in root
            ]

        if not quiet:
            print("Parsing complete.")

        return data

    def _preprocess_question(self, question: str) -> str:
        tokenized = self._tokenizer(
            question,
            add_special_tokens=False,
        )

        assert isinstance(
            tokenized["input_ids"], list
        ), f"{type(tokenized['input_ids']).mro() = }"
        assert isinstance(
            tokenized["input_ids"][0], int
        ), f"{type(tokenized['input_ids'][0]).mro() = }"

        return self._tokenizer.decode(
            tokenized["input_ids"],
            skip_special_tokens=True,
        ).strip()

    def __len__(self) -> int:
        return len(self._ds)

    def _get_indiv_item(self, idx: int):
        return dict(
            question=self._preprocess_question(self._ds[idx]["question"]),
            answer=self._ds[idx]["answer"],
        )

    def __getitem__(self, idx_or_slice: typing.Union[int, slice]):
        if isinstance(idx_or_slice, int):
            return self._get_indiv_item(idx_or_slice)

        elif isinstance(idx_or_slice, slice):
            return [
                self._get_indiv_item(i)
                for i in range(
                    idx_or_slice.start,
                    idx_or_slice.stop,
                    idx_or_slice.step,
                )
            ]


class GSM8K(torch.utils.data.Dataset):
    _int_patt = re.compile(r"\-?\d+")

    def __init__(
        self,
        *,
        max_length: int,
        tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
        device: torch.device,
        ds: collections.abc.Sequence[str],
        question_prefix: str,
        question_suffix: str,
    ):
        self._question_prefix = question_prefix
        self._ref_scratchpads = []
        self._question_suffix = question_suffix
        self._ref_equations = []
        self._outputs_key = "answer"
        self._ref_answers = []
        self._inputs_key = "question"
        self._max_length = max_length
        self._tokenizer = tokenizer
        self._input_ids = []
        self._queries = []
        self._device = device
        self._populate_ds(ds)

    def _populate_ds(self, ds):
        queries = []
        scratchpads = []
        responses = []

        for idx in range(len(ds)):
            sample = ds[idx][self._inputs_key].strip()
            scratchpad, answer = ds[idx][self._outputs_key].split("####")

            scratchpad = scratchpad.strip()
            answer = answer.strip().replace(",", "")

            if str(int(answer)) != answer.strip():
                assert False, f"{answer = }"

            queries.append(self._question_prefix + sample + self._question_suffix)
            scratchpads.append(scratchpad)
            responses.append(answer)

        LOGGER.info("> Tokenizing.")

        tokenized_queries = [
            torch.tensor(x, dtype=torch.long)
            for x in self._tokenizer(queries)["input_ids"]
        ]

        tokenized_ref_answers = [
            torch.tensor(x, dtype=torch.long)
            for x in self._tokenizer(responses)["input_ids"]
        ]

        tokenized_scratchpads = [
            torch.tensor(x, dtype=torch.long)
            for x in self._tokenizer(scratchpads)["input_ids"]
        ]

        detokenized_queries = self._tokenizer.batch_decode(tokenized_queries)
        detokenized_ref_answers = self._tokenizer.batch_decode(tokenized_ref_answers)
        detokenized_ref_scratchpad = self._tokenizer.batch_decode(tokenized_scratchpads)

        LOGGER.info("< Done Tokenizing.")

        for r_s, t_q, dt_q, dt_ra, dt_rs in more_itertools.zip_equal(
            scratchpads,
            tokenized_queries,
            detokenized_queries,
            detokenized_ref_answers,
            detokenized_ref_scratchpad,
        ):
            if len(t_q) <= self._max_length:
                # Extract the insides of the equation annotations.
                splitted = re.findall(r"<<[\(\)0-9\+\-/\*=\.]+>>", r_s)  # type: ignore
                count_left_side = r_s.count("<<")  # type: ignore
                assert len(splitted) == count_left_side, (
                    len(splitted),
                    count_left_side,
                    r_s,
                    splitted,
                )

                for eqn_str in splitted:
                    assert eqn_str[:2] == "<<", f"`{eqn_str}`"
                    assert eqn_str[-2:] == ">>", f"`{eqn_str}`"
                    eqn_str = eqn_str[2:-2]
                    left, answer = eqn_str.split("=")
                    self._ref_equations.append(
                        dict(
                            left=left,
                            answer=answer,
                        )
                    )

                self._input_ids.append(t_q)
                self._queries.append(dt_q)
                self._ref_answers.append(dt_ra)
                self._ref_scratchpads.append(dt_rs)

        LOGGER.info(
            f"[red bold]With len {self._max_length} - "
            f"Kept {len(self._queries)  /  len(queries):0.1%} samples, "
            f"{     len(self._queries)} / {len(queries)}"
        )

    def __len__(self):
        assert len(self._queries) == len(
            self._ref_answers
        ), f"{len(self._queries) = }, {len(self._ref_answers) = }"
        return len(self._queries)

    def __getitem__(
        self, idx_or_slice: typing.Union[int, slice]
    ) -> typing.Union[str, typing.Any]:
        return dict(
            query=self._queries[idx_or_slice],
            input_ids=self._input_ids[idx_or_slice],
            ref_answer=self._ref_answers[idx_or_slice],
            ref_scratchpad=self._ref_scratchpads[idx_or_slice],
            ref_equation=self._ref_equations[idx_or_slice],
        )


class ArithmeticDummyDS:
    """

    This is a terrible hack just to test with the previous project's data
    out of the box. We aren't using this data in the actual project.

    """

    def __init__(self, ds, tokenizer):
        self._ds = ds
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        assert False
        return {
            "inputs": self._tokenizer.decode(
                self._ds[idx]["input"], skip_special_tokens=True
            ),
            "labels": self._tokenizer.decode(
                self._ds[idx]["value"], skip_special_tokens=True
            ),
        }


class DatasetChoices(str, enum.Enum):
    ASDIV = "asdiv"
    GSM8K = "gsm8k"
    ARITHMETIC_DUMMY = "arithmetic_dummy"


DATASET_KEY_TO_CLASS = {
    DatasetChoices.ASDIV: ASDiv,
    DatasetChoices.GSM8K: GSM8K,
    DatasetChoices.ARITHMETIC_DUMMY: ArithmeticDummyDS,
}


def prep_dataset(
    *,
    input_max_length: int,
    question_prefix: str,
    question_suffix: str,
    task_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    split: str,
) -> torch.utils.data.Dataset:
    if task_name == lib_utils.Task.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        dataset = GSM8K(
            max_length=input_max_length,
            tokenizer=tokenizer,
            device=torch.device(LOCAL_RANK),
            ds=datasets.load_dataset(  # type: ignore
                split=split,
                path="gsm8k",
                name="main",
            ),
            question_prefix=question_prefix,
            question_suffix=question_suffix,
        )

    elif task_name == "asdiv":
        assert split is None, "split must be None for ASDiv"
        dataset = ASDiv(
            tokenizer=tokenizer,
            cache_path="/tmp/asdiv",
        )

    elif task_name == lib_utils.Task.SENTIMENT:
        dataset = lib_sentiment_specific.prep_dataset(
            txt_in_len=5,
            tokenizer=tokenizer,
            split=split,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return dataset
