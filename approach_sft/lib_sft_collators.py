import more_itertools as mit
import numpy as np
import transformers
import torch

from with_trl import lib_base_classes
import approach_sft.lib_sft_constants as lib_sft_constants


class EncoderDecoderCollator:
    def __init__(self, output_type, tokenizer):
        assert False

        self._output_type = output_type
        self._tokenizer = tokenizer
        self._data_collator_base = transformers.DataCollatorForSeq2Seq( # type: ignore
            tokenizer=tokenizer
        )
        raise NotImplementedError("TODO: implement this collator")

    @property
    def output_type(self):
        return self._output_type

    @property
    def tokenizer(self):
        return self._tokenizer

    def __call__(self, features):
        output_features = dict(text=features["query"])

        if self.output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY:
            output_features["labels"] = features["ref_answer"]

        elif (
            self.output_type
            == lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER
        ):
            output_features["labels"] = features["ref_scratchpad"]

        else:
            raise self.output_type

        return self._data_collator_base(output_features)


class CausalFullCollator:
    def __init__(
        self, 
        *, 
        output_type, 
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
    ):
        assert False
        self._forward_tokenizer = forward_tokenizer
        self._prediction_tokenizer = prediction_tokenizer
        self._output_type = output_type
         
        assert self._prediction_tokenizer.padding_side == "left", (
            self._prediction_tokenizer.padding_side)
        assert self._forward_tokenizer.padding_side == "right", (
            self._forward_tokenizer.padding_side)

    @property
    def output_type(self):
        return self._output_type

    def __call__(self, features: list[lib_base_classes.DataItemContainer]):
        """
        Two main modes:
        - Chain of thought then answer
        - Answer only
        """
        assert False

        if self.output_type == lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER:
            questions   = [f["ref_qa_question"].strip() for f in features]
            answers     = [f["ref_qa_answer"].strip() for f in features]
            choices     = [f["ref_qa_choices"].strip() for f in features]

            generations = [f["output"].strip() for f in features]

            full_query_text = [
                f"Q: {question}\n" +
                f"Answer Choices:\n{choice}\n"
                for question, choice 
                in mit.zip_equal(questions, choices)
            ]

            forward_input_text = [
                full_query +
                f"A: {generation}"
                for full_query, generation 
                in mit.zip_equal(full_query_text, generations)
            ]

            predict_input_text = [
                full_query +
                "A: "
                for full_query in full_query_text
            ]

        elif self.output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY:
            questions = [f["ref_qa_question"].strip() for f in features]
            choices   = [f["ref_qa_choices"].strip() for f in features]
            answers   = [f["ref_qa_answer"].strip() for f in features]

            forward_input_text = [
                f"Q: {question}\n" +
                "Answer Choices:\n" +
                f"{choice}\n" +
                f"A: {answer}"
                for question, choice, answer 
                in mit.zip_equal(questions, choices, answers)
            ]
            predict_input_text = [
                f"Q: {question}\n" +
                f"Answer Choices:\n{choice}\n" +
                "A: "
                for question, choice in mit.zip_equal(questions, choices)
            ]
            
        else:
            raise NotImplementedError(self.output_type)

        # Add an EOS token to the end of the forward input text
        forward_input_ids = [
            self._forward_tokenizer(forward_input_text).input_ids + 
            [self._forward_tokenizer.eos_token_id] 
            for forward_input_text in forward_input_text
        ]
        predict_input_ids = self._prediction_tokenizer(
            predict_input_text, padding=True, return_tensors="pt"
        )

        del forward_input_text
        assert self._forward_tokenizer.pad_token_id != self._forward_tokenizer.eos_token_id, (
            "This is bad for training with the language modeling collator. It makes it so "
            "the eos token is masked for no reason, the model doesn't know when to stop."
        )

        keys = features[0].keys()
        return dict(
            forward=self._forward_tokenizer.pad(dict(input_ids=forward_input_ids), return_tensors="pt"),
            predict=predict_input_ids,
            extra_info={key: [f[key] for f in features] for key in keys},
        )

def pad_to_max(sequence, pad_token):
    max_length = max(len(seq) for seq in sequence)
    return [seq + [pad_token] * (max_length - len(seq)) for seq in sequence]

class ArithmeticCausalMaskedCollator:
    def __init__(
        self, 
        *, 
        output_type: lib_sft_constants.OutputTypes,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        has_choices: bool,
    ):
        
        self._forward_tokenizer = forward_tokenizer
        self._prediction_tokenizer = prediction_tokenizer
        self._output_type = output_type
        self._has_choices = has_choices

        assert self._prediction_tokenizer.padding_side == "left", (
            self._prediction_tokenizer.padding_side)
        assert self._forward_tokenizer.padding_side == "right", (
            self._forward_tokenizer.padding_side)

    @property
    def output_type(self):
        return self._output_type

    def __call__(self, features: list[lib_base_classes.DataItemContainer]):
        """
        Two main modes:
        - Chain of thought then answer
        - Answer only
        """

        questions   = [f["ref_qa_question"].strip() for f in features]
        answers     = [f["ref_qa_answer"  ].strip() for f in features]
        scratchpads = [f["ref_qa_scratchpad"].strip() for f in features]

        if self.output_type == lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER:
            forward_input = [
                f"{q} = {s}{self._forward_tokenizer.eos_token}"
                for q, s in 
                mit.zip_equal(questions, scratchpads)
            ]

            forward_input = self._forward_tokenizer(
                forward_input, 
                padding=True,
                return_offsets_mapping=True,
            )

            pred_inputs = [f"{q} =" for q in questions]
            pred_inputs = self._forward_tokenizer(
                pred_inputs, 
                return_offsets_mapping=True,
            )

        elif self.output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY:
            forward_inputs = [
                f"{q} = {a}{self._forward_tokenizer.eos_token}"
                for q, a in 
                mit.zip_equal(questions, answers)
            ]

            forward_inputs = self._forward_tokenizer(
                forward_inputs, 
                return_offsets_mapping=True,
            )

            pred_inputs = [f"{q} =" for q in questions]
            pred_inputs = self._forward_tokenizer(
                pred_inputs, 
                return_offsets_mapping=True,
            )

            # Functional mask code: In practice this is useless
            # Find the "=" token, and mask up to and including that token
            # We find the "=" token with bisect and the offset mapping
            # We mask the "=" token with -100
            # inputs = self._forward_tokenizer(
            #     forward_input, padding=True, return_offsets_mapping=True)
            # masked_forward = inputs.input_ids.copy()
            # first_offset = [
            #     [pair[0] for pair in offset_sequence] 
            #     for offset_sequence in inputs.offset_mapping
            # ]

            # # We pad the first offset to the max length of the sequence to keep
            # # things sorted
            # first_offset = pad_to_max(first_offset, np.iinfo(first_offset[0][0]).max)
            # first_offset = torch.tensor(first_offset)
            # equal_position = torch.tensor([[
            #     text_input.find("=")] for text_input in forward_input
            # ])
            # pos = torch.searchsorted(first_offset, equal_position)
            # # Mask up to and including the "=" token
            
            # padded_inputs = self._forward_tokenizer.pad(
            #     dict(input_ids=inputs.input_ids), 
            #     return_tensors="pt",
            # ).input_ids
            # masked_inputs = padded_inputs.input_ids.clone()
            # indices = torch.arange(padded_inputs.shape[1]).unsqueeze(0)
            # mask = indices < pos
            # masked_inputs[mask] = 0

        else:
            raise NotImplementedError(self.output_type)

        forward_input_tok = self._forward_tokenizer.pad(
            dict(input_ids=forward_inputs.input_ids), 
            return_tensors="pt",
        )
        assert self._forward_tokenizer.eos_token_id is not None, "eos_token_id is None?"
        qty_unmasked_eos = (
            forward_input_tok.input_ids == self._forward_tokenizer.eos_token_id
        ).logical_and(forward_input_tok.attention_mask).sum(-1)
        assert (qty_unmasked_eos == 1).all(), qty_unmasked_eos

        predict_input_tok = self._prediction_tokenizer.pad(
            dict(input_ids=pred_inputs.input_ids), 
            return_tensors="pt",
        )
        assert self._prediction_tokenizer.eos_token_id is not None, "eos_token_id is None?"
        qty_unmasked_eos = (
            predict_input_tok.input_ids == self._prediction_tokenizer.eos_token_id
        ).logical_and(predict_input_tok.attention_mask).sum(-1)
        assert (qty_unmasked_eos == 0).all(), qty_unmasked_eos


        # masked_forward = self._forward_tokenizer.pad(
        #     dict(input_ids=masked_forward), 
        #     return_tensors="pt",
        # )

        # assert masked_forward.input_ids.shape == forward_input_tok.input_ids.shape, (
        #     masked_forward.input_ids.shape, forward_input_tok.input_ids.shape,
        # )

        keys = features[0].keys()

        return dict(
            forward        = forward_input_tok,
            # masked_forward = masked_forward,
            predict        = predict_input_tok,
            extra_info     = {key: [f[key] for f in features] for key in keys},
        )

