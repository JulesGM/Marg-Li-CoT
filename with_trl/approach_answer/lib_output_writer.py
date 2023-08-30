   
import itertools as it
import pathlib
import typing
from typing import Any, Optional, Union

import h5py
import numpy as np
import rich
import torch
import transformers


class OutputSampleWriter:
    WANTED_KEYS = dict(
        ref_qa_id_txt="ref_qa_id_detok", 
        ref_qa_question_txt="ref_qa_question_detok", 
        ref_qa_choices_txt="ref_qa_choices_detok", 
        ref_qa_answer_txt="ref_qa_answer_detok",
    )

    def __init__(
        self, 
        *, 
        do_distillation: bool,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        dataset_split_size: int,
        output_path: str, 
        split: str,
        max_gen_len: int,
        few_shots_str,
    ):
        assert split in ("train", "eval",)
        self._split = split
        self._output_path = pathlib.Path(output_path)
        self._prediction_tokenizer = prediction_tokenizer
        self._forward_tokenizer = forward_tokenizer
        self._dataset_split_size = dataset_split_size
        self._max_gen_len = max_gen_len
        self._do_distillation = do_distillation

        self._setup_h5_dataset(few_shots_str)

        forward_tokenizer.save_pretrained(self._output_path / "forward_tokenizer")
        prediction_tokenizer.save_pretrained(self._output_path / "prediction_tokenizer")
        
    def _setup_h5_dataset(self, few_shots_str):
        """
        Create the HDF5 file & associated variables.

        It has:
        - input_ids: (dataset_split_size, max_gen_len) int32, defaults to the padding token
        - logits: (dataset_split_size, max_gen_len, vocab_size) float32, defaults to NaN
        - 
        
        """
        assert not hasattr(self, "_output_h5py"), (
            "HDF5 file already created. _setup_h5_dataset is meant to be called only once.")
        self._output_h5 = h5py.File(self._output_path / f"samples.{self._split}.h5py", "w")
        self._output_h5.attrs["few_shots_str"] = few_shots_str

        rich.print("-> Creating `clean_gen_ids`")
        self._output_h5.create_dataset("clean_gen_ids",
            shape=(self._dataset_split_size, self._max_gen_len), 
            dtype=np.int32)
        self._output_h5["clean_gen_ids"][:] = self._prediction_tokenizer.pad_token_id
        rich.print("-> Creating `clean_gen_logits`")
        
        if self._do_distillation:
            self._output_h5.create_dataset("clean_gen_logits",
                shape=(self._dataset_split_size, self._max_gen_len, self._prediction_tokenizer.vocab_size), 
                dtype=np.float32,)
            self._output_h5["clean_gen_logits"][:] = float("nan")

        rich.print("-> Creating `ref_qa_question_txt`")
        self._output_h5.create_dataset("ref_qa_question_txt", shape=(self._dataset_split_size,), dtype=h5py.string_dtype())
        rich.print("-> Creating `ref_qa_id_txt`")
        self._output_h5.create_dataset("ref_qa_id_txt",       shape=(self._dataset_split_size,), dtype=h5py.string_dtype())
        rich.print("-> Creating `ref_qa_choices_txt`")
        self._output_h5.create_dataset("ref_qa_choices_txt",  shape=(self._dataset_split_size,), dtype=h5py.string_dtype())
        rich.print("-> Creating `ref_qa_answer_txt`")
        self._output_h5.create_dataset("ref_qa_answer_txt",   shape=(self._dataset_split_size,), dtype=h5py.string_dtype())
        self._h5py_idx = 0
        rich.print("-> Done creating HDF5 file.")

    def close(self):
        self._output_h5.close()

    def __call__(
        self, 
        *, 
        gathered_batch,
        clean_gathered_output_sample_ids,
        clean_gathered_logits,
    ) -> Any:

        assert len(gathered_batch) == len(clean_gathered_output_sample_ids), (
            len(gathered_batch), len(clean_gathered_output_sample_ids))
        received_batch_size = len(gathered_batch)

        clean_seq_len = clean_gathered_output_sample_ids.shape[1]
        assert clean_seq_len <= self._max_gen_len, (clean_seq_len, self._max_gen_len)
        upper_bound_bsz = min(received_batch_size, self._output_h5["clean_gen_ids"].shape[0] - self._h5py_idx)
        
        
        self._output_h5["clean_gen_ids"][
            self._h5py_idx:self._h5py_idx + upper_bound_bsz,
            :clean_seq_len
        ] = clean_gathered_output_sample_ids[:upper_bound_bsz]
        
        if self._do_distillation:
            assert len(gathered_batch) == len(clean_gathered_logits), (
                len(gathered_batch), len(clean_gathered_logits))
            assert clean_seq_len == clean_gathered_logits.shape[1], (
                clean_seq_len, clean_gathered_logits.shape[1], clean_gathered_output_sample_ids.shape[1])
            assert len(gathered_batch) == clean_gathered_logits.shape[0], (
                len(gathered_batch), clean_gathered_logits.shape[0])
            
            self._output_h5["clean_gen_logits"][
                self._h5py_idx:self._h5py_idx + upper_bound_bsz,
                :clean_seq_len
            ] = clean_gathered_logits[:upper_bound_bsz]
        
       
        for i, sample in enumerate(gathered_batch):
            write_position = self._h5py_idx + i

            if write_position >= self._output_h5["clean_gen_ids"].shape[0]:
                break

            # Dump other keys to the h5 file
            for k_h5, v_batch in self.WANTED_KEYS.items():
                self._output_h5[k_h5][write_position] = sample[v_batch]

        self._h5py_idx += upper_bound_bsz

    @classmethod
    def gather_batch_for_writing(cls, local_batch):
        only_needed_keys = []
        
        for sample in local_batch:
            only_needed_keys.append({
                k: sample[k] for k in cls.WANTED_KEYS.values()
            })

        output = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(output, only_needed_keys)

        return list(it.chain.from_iterable(output))
