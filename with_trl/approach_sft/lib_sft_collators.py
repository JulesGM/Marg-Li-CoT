import more_itertools as mit
import transformers

import lib_base_classes
import lib_data
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

        if self.output_type == lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER:
            questions = [f["ref_qa_question"].strip() for f in features]
            answers = [f["ref_qa_answer"].strip() for f in features]
            choices = [f["ref_qa_choices"].strip() for f in features]
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
                f"A: "
                for full_query in full_query_text
            ]

        elif self.output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY:
            questions = [f["ref_qa_question"].strip() for f in features]
            answers = [f["ref_qa_answer"].strip() for f in features]

            forward_input_text = [
                f"Q: {question}\n" +
                f"Answer Choices:\n" +
                f"{choices}\n" +
                f"A: {answer}"
                for question, answer 
                in mit.zip_equal(questions, answers)
            ]
            predict_input_text = [
                f"Q: {question}\n" +
                f"Answer Choices:\n{choices}\n" +
                f"A: "
                for question in questions
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

