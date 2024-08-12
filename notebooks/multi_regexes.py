import outlines
import outlines.models.transformers
import outlines.samplers
from outlines.generate.generator import sequence_generator

from typing import *

class MultiFSMSequenceGenerator(
    outlines.generate.api.SequenceGenerator
):
    def __init__(
        self,
        model,
        sampler,
        device,
    ):
        self.model = model
        self.sampler = sampler
        self.tokenizer = model.tokenizer
        self.device = device
        self.num_samples = sampler.samples

    def __call__(
        self,
        prompts: Union[str, List[str]],
        fsms,
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional["torch.Generator"] = None,
    ) -> Union[
        outlines.generate.api.FormattedOutput, List[
            outlines.generate.api.FormattedOutput], List[List[
                outlines.generate.api.FormattedOutput]]]:
        """Generate the full text sequence.

        Since `SequenceGenerator.stream` calls the tokenizer at every step this
        method loops over the generator returned by `sequence_generator` itself
        so the tokenizer is called only once after all token ids have been
        generated.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        The generation(s), potentially cast to another type.
        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at
        num_samples = self.num_samples

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_samples
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks  = torch.repeat_interleave(attention_masks , num_samples, dim=0)
        fsm_states       = [0 for _ in range(batch_size * num_samples)]
        
        if isinstance(fsms, (list, tuple)):
            assert len(fsms) == batch_size, (
                f"{len(fsms) = }, {batch_size = }")

            copied_fsms = []
            for i in range(batch_size):
                for _ in range(num_samples):
                    copied_fsms.append(fsms[i].copy())

        else:
            copied_fsms = [fsms.copy() for _ in range(batch_size * num_samples)]

        weights = torch.zeros(
            (batch_size * num_samples), dtype=torch.float, device=self.device
        )

        states = sequence_generator(
            self.model,
            self.sampler,
            copied_fsms,
            prompt_token_ids,
            weights,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        while True:
            try:
                last_state = next(states)
                if max_tokens or stop_sequences:
                    token_ids = last_state.token_ids
                    generated_token_ids = self.get_generated_token_ids(
                        prompt_token_ids, token_ids
                    )
                    if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                        break
                    if stop_sequences and self.is_stop_sequence_found(
                        self.tokenizer.decode(generated_token_ids), stop_sequences
                    ):
                        break
            except StopIteration:
                break

        token_ids = last_state.token_ids
        generated_token_ids = self.get_generated_token_ids(
            prompt_token_ids, token_ids)

        generated = self.tokenizer.decode(generated_token_ids)
        stripped = [
            self.strip_stop_sequences(sequence, stop_sequences)
            for sequence in generated
        ]
        formatted = [self.format_sequence(sequence) for sequence in stripped]

        # We reshape the output to (batch_size, sample_size)
        output: List[List[outlines.generate.api.FormattedOutput]] = list()
        for i in range(0, batch_size * num_samples, num_samples):
            output.append(formatted[i : i + num_samples])

        # We remove leading dimensions for the output
        if batch_size == 1 and num_samples == 1:
            return output[0][0]
        elif batch_size == 1:
            return output[0]
        elif num_samples == 1:
            return [samples[0] for samples in output]
        else:
            return output


class MultiRegexGenerator:
    def __init__(self, model, sampler):
        self._seq_gen = MultiFSMSequenceGenerator(
            model=model, 
            sampler=sampler, 
            device=model.model.device,
        )

    def __call__(self, prompts, regexes_str, **kwargs):
        fsms = []
        for regex_str in regexes_str:
            new_fsm = outlines.fsm.guide.RegexGuide(
                regex_str, 
                self._seq_gen.model.tokenizer,
            )
            new_fsm.regex_str = regex_str
            fsms.append(new_fsm)

        return self._seq_gen(prompts, fsms, **kwargs)


def print_states(fsm, tokenizer):
    for state, token_to_state in fsm.states_to_token_maps.items():
        print(f"state: {state}")
        print({
            tokenizer.tokenizer.decode([token]): target_state 
            for token, target_state in token_to_state.items()
        })