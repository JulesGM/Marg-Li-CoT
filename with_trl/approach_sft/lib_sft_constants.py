import enum


class LMModes(enum.Enum):
    """

    SEQ2SEQ: The seq2seq model is trained to generate the output, given the input, in the
    regular seq2seq fashion.
    
    CAUSAL_CONDITIONAL: The causal model is trained to generate the output, given the input. 
    It is not trained to generate the input as well.

    CAUSAL_FULL: The model is trained to generate the input and the output.

    """
    CAUSAL_CONDITIONAL = "causal_conditional"
    CAUSAL_FULL = "causal_full"


class OutputTypes(enum.Enum):
    ANSWER_ONLY = "answer_only"
    CHAIN_OF_THOUGHT_THEN_ANSWER = "chain_of_thought_then_answer"


class DataModes(enum.Enum):
    OPENAI_GENERATION = "openai_generation"
    H5_GENERATION = "h5_generation"

