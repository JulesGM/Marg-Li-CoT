
import re
import bisect
import random

import more_itertools
import rich
import torch
import transformers


def find_intermediate_answers(text: str) -> list[tuple[int, int]]:
    """ In the Arithmetic dataset, find the answer.

    Args:
        text (str): The text to search in.
    Returns:
        list[tuple[int, int]]: The start and end of the answer.
    """

    spans = []
    for match in re.finditer(r",( (\d )*\d )", text):
        spans.append(match.span(1))
    return spans


def string_span_to_input_ids_span(span_str: tuple[int, int], idx_offset_mapping: list[int]) -> tuple[int, int]:
    """ Extract the id indicies associated to a string span.
    """

    assert isinstance(span_str, (tuple, list)), type(span_str).mro()
    assert len(span_str) == 2, len(span_str)
    assert span_str[0] < span_str[1], span_str

    start_bisect_idx = bisect.bisect(idx_offset_mapping, span_str[0])
    end_bisect_idx = bisect.bisect(idx_offset_mapping, span_str[1])

    return start_bisect_idx, end_bisect_idx


################################################################################################################
################################################################################################################

class ReplaceWithRandom:
    def __init__(self, good_ids):
        self._good_ids = good_ids

    def __call__(
        self,
        output_input_ids: list[int],
        bisect_idx_start: int,
        bisect_idx_end: int,
        tokenizer: transformers.PreTrainedTokenizerFast,
        model: transformers.PreTrainedModel,
    ) -> list[int]:

        assert bisect_idx_end   >  bisect_idx_start, (bisect_idx_end, bisect_idx_start)
        assert bisect_idx_start >=                0, (bisect_idx_end, bisect_idx_start)
        assert bisect_idx_end   >                 0, (bisect_idx_end, bisect_idx_start)
        output_input_ids = output_input_ids.copy()

        new_ids = [
            random.choice(self._good_ids) for _ in range(bisect_idx_end - bisect_idx_start)
        ]
        output_input_ids[bisect_idx_start:bisect_idx_end] = new_ids

        return output_input_ids, tokenizer.decode(new_ids)

class ReplaceWithMostLikely:
    def __init__(self, prefix_allowed_tokens_fn):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(
        self,
        output_input_ids: list[int],
        bisect_idx_start: int,
        bisect_idx_end: int,
        tokenizer: transformers.PreTrainedTokenizerFast,
        model: transformers.PreTrainedModel,
    ) -> list[int]:
        """
        
        https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/text_generation#transformers.GenerationMixin.generate.prefix_allowed_tokens_fn
        
        prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]], optional):
        If provided, this function constraints the beam search to allowed tokens only at 
        each step. If not provided no constraint is applied. This function takes 2 arguments: 
        the batch ID batch_id and input_ids. It has to return a list with the allowed tokens 
        for the next generation step conditioned on the batch ID batch_id and the previously 
        generated tokens inputs_ids. This argument is useful for constrained generation 
        conditioned on the prefix, as described in Autoregressive Entity Retrieval.
        
        """

        assert bisect_idx_end   >  bisect_idx_start, (bisect_idx_end, bisect_idx_start)
        assert bisect_idx_start >=                0, (bisect_idx_end, bisect_idx_start)
        assert bisect_idx_end   >                 0, (bisect_idx_end, bisect_idx_start)
        len_ = bisect_idx_end - bisect_idx_start
        output_input_ids = output_input_ids.copy()

        model_input = dict(
            input_ids=torch.tensor(
                [output_input_ids[:bisect_idx_start]], 
                dtype=torch.long,
            ).to(model.device),
        )
        model_input["attention_mask"] = torch.ones_like(model_input["input_ids"])

        noise_generation_model_input = tokenizer.decode(more_itertools.one(model_input["input_ids"]))

        rich.print(f"[bold]noise_generation_model_input:[/] \"{noise_generation_model_input}\"")

        new_ids = more_itertools.one(model.generate(
            **model_input,
            max_new_tokens=len_,
            min_new_tokens=len_,
            num_beams=2,
            do_sample=True,
            prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
        ).tolist())[bisect_idx_start:bisect_idx_end]

        rich.print(f"[bold]generated_answers:[/] \"{tokenizer.decode(new_ids) = }\"")

        assert len(new_ids) == len_, (len(new_ids), len_)
        output_input_ids[bisect_idx_start:bisect_idx_end] = new_ids
        
        return output_input_ids, tokenizer.decode(new_ids)