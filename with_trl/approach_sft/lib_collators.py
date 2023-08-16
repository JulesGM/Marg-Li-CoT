
import transformers

import lib_base_classes
import lib_data
from libs_sft import lib_constants


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

        if self.output_type == lib_constants.OutputTypes.ANSWER_ONLY:
            output_features["labels"] = features["ref_answer"]

        elif (
            self.output_type
            == lib_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER
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
        forward_tokenizer, 
        prediction_tokenizer,
    ):

        self._forward_tokenizer = forward_tokenizer
        self._prediction_tokenizer = prediction_tokenizer

        self._output_type = output_type
        self._forward_inner_collator = (
            transformers.DataCollatorForLanguageModeling( # type: ignore
                tokenizer=forward_tokenizer, 
                mlm=False,
            ))
        
        assert self._prediction_tokenizer.padding_side == "left", (
            self._prediction_tokenizer.padding_side)
        assert self._forward_tokenizer.padding_side == "right", (
            self._forward_tokenizer.padding_side)

    @property
    def output_type(self):
        return self._output_type

    def __call__(self, features: list[lib_base_classes.DataItemContainer]):

        if self.output_type == lib_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER:
            forward_input_text = [
                # self._forward_tokenizer.bos_token +
                self._forward_tokenizer.decode(f.tok_ref_query, skip_special_tokens=True) + " " +
                f["output"].strip()
                for f in features
            ]

        elif self.output_type == lib_constants.OutputTypes.ANSWER_ONLY:
            forward_input_text = [
                # self._forward_tokenizer.bos_token +
                self._forward_tokenizer.decode(f.tok_ref_query, skip_special_tokens=True) +  
                self._forward_tokenizer.decode(f.tok_ref_answer, skip_special_tokens=True)
                for f in features
            ]
            
        else:
            raise NotImplementedError(self.output_type)

        forward_input_ids = [
            self._forward_tokenizer(forward_input_text).input_ids + 
            [self._forward_tokenizer.eos_token_id] 
            for forward_input_text in forward_input_text
        ]

        del forward_input_text
        assert self._forward_tokenizer.pad_token_id != self._forward_tokenizer.eos_token_id, (
            "This is bad for training with the language modeling collator. It makes it so "
            "the eos token is masked for no reason, the model doesn't know when to stop."
        )

        return dict(
            forward=self._forward_inner_collator(forward_input_ids),
            predict=lib_data.data_item_collator(features)
        )

