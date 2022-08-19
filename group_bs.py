# coding=utf-8
import math
import openai
import warnings

import torch
import torch.distributed as dist
from torch import nn

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    LogitsProcessorList,
    HammingDiversityLogitsProcessor,
    MinLengthLogitsProcessor,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_utils import BeamSearchDecoderOnlyOutput
from transformers.pytorch_utils import torch_int_div

import multiprocessing.pool as mp_pool
import tqdm

def gpt3_one_step(context, num_lprobs=100):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=context,
        suffix=None,
        max_tokens=1,
        temperature=1.0,
        logprobs=num_lprobs
    )
    
    token = response["choices"][0]["text"]
    lprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
    return token, lprobs


def gpt3_generate(input_ids, tokenizer, is_parallel: bool):
    """
    Args:
        input_ids ([batch_size * beam_size, seq_len])

    Returns:
        lprobs ([batch_size * beam_size, voacb_size])

    """
    assert input_ids.dim() == 2
    batch_size, seq_len = input_ids.shape

    outputs = torch.zeros(batch_size, len(tokenizer)).to('cuda')
    def fn(idx):
        input_text = tokenizer.decode(input_ids[idx])
        _, lprobs = gpt3_one_step(input_text)

        for k, v in lprobs.items():
            outputs[idx, tokenizer.encode(k)[0]] = math.exp(v)

    jobs = []
    if is_parallel:
        with mp_pool.ThreadPool(batch_size) as pool:
            jobs.extend([pool.apply_async(fn, (i,)) for i in range(batch_size)])

            # Join, essentially
            # [job.get() for job in tqdm.tqdm(jobs, desc="Generating.")]
            [job.get() for job in jobs]
    else:
        [fn(i) for i in range(batch_size)]

    outputs = outputs / outputs.sum(dim=-1, keepdim=True)
    return outputs


def expand_inputs_for_generation(
    input_ids,
    expand_size = 1,
    attention_mask = None,
    **model_kwargs
):
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    return input_ids, model_kwargs


def group_beam_search(
    tokenizer,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    parallel_generation: bool = False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **diverse beam search
    decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

        model_kwargs:
            Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation_utils.BeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if [`~generation_utils.BeamSearchDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
        [`~generation_utils.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.

    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    output_attentions = False
    output_hidden_states = False
    return_dict_in_generate = False

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    device = input_ids.device

    batch_beam_size, cur_len = input_ids.shape

    if return_dict_in_generate and output_scores:
        beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
    else:
        beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
    # the same group don't produce same tokens everytime.
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # predicted tokens in cur_len step
        current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

        # do one decoder step on all beams of all sentences in batch
        # model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = gpt3_generate(input_ids, tokenizer, parallel_generation)

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        if output_scores:
            processed_score = torch.zeros_like(outputs)

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of current group only
            next_token_logits = outputs[batch_group_indices, :]

            # next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            # next_token_scores = nn.functional.log_softmax(
            #     next_token_logits, dim=-1
            # )  # (batch_size * group_size, vocab_size)
            next_token_scores = torch.log(
                next_token_logits
            )  # (batch_size * group_size, vocab_size)
            vocab_size = next_token_scores.shape[-1]

            next_token_scores_processed = logits_processor(
                group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
            )
            next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            if output_scores:
                processed_score[batch_group_indices] = next_token_scores_processed

            # reshape for beam search
            next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if return_dict_in_generate and output_scores:
                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                )

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            current_tokens[batch_group_indices] = group_input_ids[:, -1]

            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch_int_div(beam_idx, group_size) + group_start_idx + (beam_idx % group_size)
            )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        # model_kwargs = model._update_model_kwargs_for_generation(
        #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        # )
        # if model_kwargs["past"] is not None:
        #     model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], reordering_indices)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        return BeamSearchDecoderOnlyOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            beam_indices=sequence_outputs["beam_indices"],
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        )
    else:
        return sequence_outputs["sequences"]


def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
    """
    Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
    """
    return {"input_ids": input_ids}