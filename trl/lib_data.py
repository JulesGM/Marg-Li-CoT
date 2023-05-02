import abc
import bisect
import collections
import enum
import logging
import os
import re
import time
import typing
import xml
from pathlib import Path


import more_itertools
import numpy as np
import rich
import rich.box
import rich.table
import torch
import transformers
import wget
from text2digits import text2digits

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
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

    def get(self):
        if self._size == 0:
            raise ValueError("No data in the moving average window. self._size == 0")
        return self._window.sum() / self._size, (self._window.sum(), self._size)


class ConvToNum:
    def __init__(self, failure_rate_moving_average_window_size: int = 10000):
        self._failure_rate   = MovingAverage(failure_rate_moving_average_window_size)
        self._change_rate    = MovingAverage(failure_rate_moving_average_window_size)
        self._no_answer_rate = MovingAverage(failure_rate_moving_average_window_size)
        self._converter_inst = text2digits.Text2Digits(convert_ordinals=False)
        self._answer_pat     = re.compile(r"(\d+(,\d+)*)(\.\d+)?")

    @classmethod
    def _log_diffs(cls, *, level, initial, text, converted, final, change_stats):
        text_diff = "[on black] ".join([
            (f"[on black]{pre_conv}" if pre_conv == conv else f"[on red]{pre_conv}") 
            for pre_conv, conv in zip(text.split(), converted.split())
        ])
        converted_diff = "[on black] ".join([
            (f"[on black]{conv}" if pre_conv == conv else f"[on red]{conv}") 
            for pre_conv, conv in zip(text.split(), converted.split())
        ])
        final_diff = "".join([
            (f"[on black]{final_}" if final_ == conv else f"[on red]{final_}") 
            for final_, conv in zip(
                final, 
                converted
            )
        ])

        if change_stats:
            change_str = f"{change_stats[0]:0.1%}, {change_stats[1][0]}/{change_stats[1][1]}"
        else:
            change_str = ""

        LOGGER.log(
            level,
            f"\n[green bold]Converted words to numbers.\n" +
            (f"change_stats: {change_str}\n" if change_stats else "") +
            f"{initial}\n" +
            f"[green]-> with words ->[white]\n" +
            f"{text_diff}\n" +
            f"[green]-> with numbers ->[white]\n" +
            f"{converted_diff}\n" +
            (
                "" if converted != final else 
                f"[green]-> final ->[white]\n{final_diff}\n"
            )+ 
            "#" * 80 + "\n" 
        )

    @classmethod
    def _log_answers(cls, *, level, initial_answer, intermediate_answer, new_answer, final_answer):
        LOGGER.log(
            level,
            "\n" +
            "#" * 80 + "\n" + 
            f"[bold blue]initial_answer:[white]       \"{initial_answer}\"\n" +
            f"[bold blue]intermediate_answer:[white]  \"{intermediate_answer}\"\n" +
            f"[bold blue]new_answer:[white]           \"{new_answer}\"\n" +
            f"[bold blue]final_answer:[white]         \"{final_answer}\"\n" 
        )

    def conv_words_to_numbers(self, text: str) -> str:

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

        text_ = re.sub(r"(?P<nan>\D)\.(?P<num>\d)", "\g<nan> 0.\g<num>", text)
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

        initial_answer      = self.extract_answer(initial)
        initial_answer      = initial_answer.group(0) if initial_answer else None
        intermediate_answer = self.extract_answer(text)
        intermediate_answer = intermediate_answer.group(0) if intermediate_answer else None
        new_answer          = self.extract_answer(converted)
        new_answer          = new_answer.group(0) if new_answer else None
        final_answer        = self.extract_answer(final)
        final_answer        = final_answer.group(0) if final_answer else None

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
                    f"[red bold] No answer found in final text. {ratio:0.1%} {sum_}/{size}"
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

    def extract_answer(self, text: str) -> re.Match:
        
        output = more_itertools.last(
            re.finditer(self._answer_pat, text),
            default=None,
        )

        if output is None:
            # LOGGER.debug(f"[dark_orange bold on white]EXTRACT ANSWER FAILED: `{text}`")
            pass

        return output

    def __call__(self, text: str) -> str:
        return self.conv_words_to_numbers(text)


class ASDivRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_path,
        url="https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml",
        quiet=False,
    ):

        super().__init__()
        self._cache_path = Path(cache_path)
        self._url = url

        if not self._cache_path.exists():
            if not quiet:
                print("Downloading dataset...")
            wget.download(self._url, out=str(self._cache_path), bar=None)
            if not quiet:
                print("Download complete.")

        if not quiet:
            print("Parsing dataset...")

        with self._cache_path.open() as fp:
            root = xml.etree.ElementTree.parse(fp).getroot()[0]
            self._data = [
                {element.tag: element.text for element in x} | dict(x.items())
                for x in root
            ]

        if not quiet:
            print("Parsing complete.")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class ASDiv:
    def __init__(self, *args, **kwargs):
        
        assert False

        self._ds = ASDivRaw(*args, **kwargs)
        self._extra_info = {}

        for inner_item in self._ds:
            new_keys = {"question", "answer"}
            assert not any(k in inner_item for k in new_keys), new_keys - (
                new_keys & set(inner_item)
            )
        super().__init__()
        
    def preprocess_question(self, question: str) -> str:
        tokenized = self._tokenizer(
            question, add_special_tokens=False,
        )
        assert isinstance(tokenized["input_ids"], list), f"{type(tokenized['input_ids']).mro() = }"
        assert isinstance(tokenized["input_ids"][0], int), f"{type(tokenized['input_ids'][0]).mro() = }"

        return self._tokenizer.decode(
            tokenized["input_ids"], skip_special_tokens=True,
        ).strip()

    def __len__(self) -> int:
        return len(self._ds)

    def _get_indiv_item(self, index) -> str:
        inner_item = self._ds[index]
        sample = self.preprocess_question(
            inner_item["Body"] + " " + inner_item["Question"]
        )
        
        self._extra_info[sample] = {
            "answer": inner_item["Answer"],
            "scratchpad": inner_item["Formula"],
        } | inner_item
        

        assert isinstance(sample, str)
        return sample

    def __getitem__(self, idx_or_slice: typing.Union[int, slice]) -> typing.Union[str, list[str]]:
        if isinstance(idx_or_slice, int):
            return self._get_indiv_item(idx_or_slice)

        elif isinstance(idx_or_slice, slice):
            return [self._get_indiv_item(i) for i in range(
                idx_or_slice.start, idx_or_slice.stop, idx_or_slice.step)
            ]
        

class GSM8K:
    _int_patt = re.compile(r"\-?\d+")

    def __init__(
        self, 
        *,
        max_length: int,
        tokenizer: transformers.PreTrainedTokenizerBase, 
        ds: collections.abc.Sequence[str],
    ):

        self._outputs_key = "answer"
        self._inputs_key  = "question"
        self._max_length  = max_length
        self._tokenizer   = tokenizer
        self._populate_ds(ds)

    def _populate_ds(self, ds):
        samples = []
        outputs = []

        for idx in range(len(ds)):
            sample = self.preprocess_question(ds[idx][self._inputs_key])
            output = ds[idx][self._outputs_key].rsplit("####", 1)[1].strip()
            samples.append(sample)
            outputs.append(output.replace(",", ""))

        LOGGER.info("> Tokenizing.")
        tokenized_samples = self._tokenizer(samples)["input_ids"]
        tokenized_outputs = self._tokenizer(outputs)["input_ids"]
        LOGGER.info("< Done Tokenizing.")

        self._samples = []
        self._outputs = []
        
        for t_s, t_o in more_itertools.zip_equal(
            tokenized_samples, 
            tokenized_outputs, 
        ):
            if len(t_s) <= self._max_length:
                self._samples.append(t_s)
                self._outputs.append(t_o)

        LOGGER.info(
            f"[red bold]With len {self._max_length} - "
            f"Kept {len(self._samples)  /  len(samples):0.1%} samples, "
            f"{     len(self._samples)} / {len(samples)}"
        )

    def __len__(self):
        assert len(self._samples) == len(self._outputs), (
            f"{len(self._samples) = }, {len(self._outputs) = }")
        return len(self._samples)

    def __getitem__(
            self, idx_or_slice: typing.Union[int, slice]
        ) -> typing.Union[str, list[str]]:
        return self._samples[idx_or_slice], self._outputs[idx_or_slice]
        


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