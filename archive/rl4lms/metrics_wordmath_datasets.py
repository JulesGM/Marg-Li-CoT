import logging
import re
import typing

from text2digits import text2digits

import with_trlx.metric as metric
import rl4lms.envs.text_generation.registry as rl4lms_registry

LOGGER = logging.getLogger(__name__)


def deal_with_words(text: str) -> typing.Optional[float]:
    try:
        converted = text2digits.Text2Digits().convert(text)
    except (ValueError, ArithmeticError):
        converted = text

    output = _NUM_PAT.findall(converted)

    if not output:
        return None

    output = output[-1]

    LOGGER.debug("[bold blue]" + "#" * 80)
    LOGGER.debug(
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}",
    )
    LOGGER.debug("[bold blue]" + "#" * 80)

    return output


_NUM_PAT = re.compile(r"\d+(?:[\,\.]\d+)?")


def split_fn(generated_text: str, process_idx: int=None) -> typing.Optional[str]:
    output = deal_with_words(generated_text)

    if output is None:
        if process_idx == 0:
            LOGGER.info(
                f"[red]split_fn: no numbers found. \n"
                f"\t-> Received:[/] `{generated_text}`"
            )
        output = None
    return output


class WordMathIntScratchpadAnswerAccuracy(metric.ScratchpadAnswerAccuracy):
    def __init__(self):
        super().__init__(
            make_comparable_fn=metric.convert_to_int, 
            extract_answer_fn=split_fn,
        )


rl4lms_registry.MetricRegistry.add(
    "word_math_int_scratchpad_answer_accuracy", 
    WordMathIntScratchpadAnswerAccuracy,
)
