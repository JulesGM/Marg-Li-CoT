class WhiteListConstraint(transformers.Constraint):
    r"""
    [`Constraint`] enforcing that the next token must be one of the tokens in the white list.

    Args:
        white_list (:obj:`set` of :obj:`int`): The set of token ids to choose from.
    """

    def __init__(self, white_list: set[int]):
        super(transformers.Constraint, self).__init__()

        if not isinstance(white_list, set) or len(white_list) == 0:
            raise ValueError(f"`white_list` has to be a non-empty set, but is {white_list}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in white_list):
            raise ValueError(f"Each list in `white_list` has to be a list of positive integers, but found {[token_id for token_id in white_list if (not isinstance(token_id, int) or token_id < 0)]}.")

        self.white_list = white_list
        self.seqlen = len(self.white_list)
        self.completed = False

    def advance(self):
        if self.completed:
            return None
        return None

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int) or not token_id >= 0:
            raise ValueError(f"`token_id` has to be a positive `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False

        return token_id in self.white_list

    def update(self, token_id: int):
        if not isinstance(token_id, int) or not token_id >= 0:
            raise ValueError(f"`token_id` has to be a positive `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            stepped = True
            completed = True
            self.completed = completed
        else:
            reset = True
            self.reset()

        return stepped, completed, reset

    def reset(self):
        self.completed = False

    def remaining(self):
        return 0 if self.completed else 1

    def copy(self, stateful=False):
        new_constraint = self.__class__(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.completed = self.completed

        return new_constraint
    

    import abc
class GenAnswer(transformers.Constraint):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    def __init__(self, tokenizer):
        numbers = "0123456789"
        numbers_ids = set([v for k, v in t.vocab.items() if k in numbers and len(k) == 1])
        self._tokenizer = tokenizer
        self._number_ids = numbers_ids
        self._space_id = tokenizer.vocab["▁"]
        self._last_was_number = False
        super().__init__()


    @abc.abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        if self._last_was_number:
            self._last_was_number = False
            return torch.tensor([self._space_id], dtype=torch.long)
        
        self._last_was_number = True
        return torch.tensor(self._number_ids, dtype=torch.long)

    @abc.abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """

        
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abc.abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abc.abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abc.abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abc.abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
