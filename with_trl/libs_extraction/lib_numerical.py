import logging
import math
import re
from typing import Optional

import more_itertools
from text2digits import text2digits

from with_trl import lib_utils
from . import lib_base

LOGGER = logging.getLogger(__name__)


class ConvToNum(lib_base.Extractor):
    """Converts words to numbers.
    
    As a BaseExtractor, __call__ returns a single answer, and compare compares two answers.
    To return a single answer, all numbers are extracted, then the last one is returned.

    Keeps internal failure rate statistics.
    
    """
    def __init__(
            self, 
            failure_rate_moving_average_window_size: int = 10000,
        ):
        self._failure_rate = lib_utils.MovingAverage(failure_rate_moving_average_window_size)
        self._change_rate = lib_utils.MovingAverage(failure_rate_moving_average_window_size)
        self._no_answer_rate = lib_utils.MovingAverage(failure_rate_moving_average_window_size)
        self._converter_inst = text2digits.Text2Digits(convert_ordinals=False)  # type: ignore
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
            change_str = f"{change_stats[0]:0.1%}, {change_stats[1][0]}/{change_stats[1][1]}"
        else:
            change_str = ""

        LOGGER.log(
            level,
            "\n[green bold]Converted words to numbers.\n"
            + (f"change_stats: {change_str}\n" if change_stats else "")
            + f"{initial}\n"
            + "[green]-> with words ->[white]\n"
            + f"{text_diff}\n"
            + "[green]-> with numbers ->[white]\n"
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

        assert isinstance(text, str), type(text)
        text_ = re.sub(r"(\d+),", r"\1", text)
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

        return final

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
            intermediate_answer=intermediate_answer,
            initial_answer=initial_answer,
            final_answer=final_answer,
            new_answer=new_answer,
            level=logging.DEBUG,
        )

        self._log_diffs(
            change_stats=None,
            converted=converted,
            initial=initial,
            level=logging.DEBUG,
            final=final,
            text=text,
        )

        # We only return the text if it has changed.
        if intermediate_answer != new_answer:
            self._change_rate.update(1)
            self._log_diffs(
                change_stats=self._change_rate.get(),
                converted=converted,
                initial=initial,
                level=logging.DEBUG,
                final=final,
                text=text,
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

    def extract_numbers(self, text: str) -> Optional[list[Optional[float]]]:
        """Helper function to just return numbers instead of matches."""
        output = self.extract_number_strings(text)
        if output is None:
            return
        
        return [self.to_float(m) for m in output]

    def to_float(self, text: str) -> Optional[float]:
        try:
            # We try directly
            converted = float(text)

        except ValueError:
            try:
                # We try to remove eventual commas.
                # We only do this if the first attempt failed,
                # because the comma could be a decimal separator.
                converted = float(text.replace(",", ""))

            except ValueError:
                LOGGER.info(
                    f"[red bold]ValueError: [white]"
                    f"`{text.replace(',', '') = }` "
                    f"`{text = }` "
                )
                return None
        return converted
    
    def __call__(self, text):
        return more_itertools.last(
            self.extract_numbers(text), None
        )
    
    def compare(self, extracted_answer_a, extracted_answer_b):
        return math.isclose(extracted_answer_a, extracted_answer_b)