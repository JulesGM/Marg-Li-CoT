import collections
import itertools as it
import logging
import os
import pathlib

import datasets
import torch
import torch.backends
import torch.backends.cuda
import torch.utils
import torch.utils.data
import transformers
import transformers.utils

from approach_sft import lib_sft_collators
from approach_sft import lib_sft_constants
from approach_sft import lib_sft_commonsense_qa
import math_sft
from with_trl import lib_utils
from with_trl.libs_data import lib_arithmetic
from with_trl.libs_data import lib_gsm8k


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)


class IterableDatasetSubset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, qty):
        self._dataset = dataset
        self._qty = qty

    def __iter__(self):
        return it.islice(self._dataset, self._qty)

    def __len__(self):
        return self._qty


def _unzip_collator(batch):
    """
    Take a list of dicts and make a dict of lists out of it.
    pd.DataFrame(batch).to_dict(orient="list") does something similar.
    """

    keys = batch[0].keys()
    outputs = collections.defaultdict(list)
    
    for sample in batch:
        assert len(sample.keys()) == len(keys), (
            sample.keys(), keys)
        
        for key in keys:
            outputs[key].append(sample[key])
    
    # Convert to regular dict to remove defaultdict behavior
    outputs = dict(outputs.items())
    
    # Ensure all values have the same length
    one_len = len(next(iter(outputs.values())))
    
    lengths = {k: len(v) for k, v in outputs.items()}
    assert all(v == one_len for k, v in lengths.items()), (lengths, one_len)
    
    return outputs


def get_num_cpus():
    """
    On SLURM, os.cpu_count returns the number of CPUs per node, not the total number of CPUs.
    os.sched_getaffinity(0) returns the number of cpus available to the current process, so 
    that's what we want, but it's not available on Windows or Mac. We don't actually care
    about those platforms, but os.cpu_count is available on them & it's low effort to add it, 
    so we use that as a fallback.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count()


def get_dataloaders(
    *,
    answer_only:                bool,
    data_directory:             pathlib.Path,
    dataset_choice:             lib_utils.Datasets,
    eval_batch_size:            int,
    extractor_ignore_one_line:  callable,
    filter_bads:                bool,
    forward_tokenizer:          transformers.PreTrainedTokenizerBase,  # type: ignore
    prediction_tokenizer:       transformers.PreTrainedTokenizerBase,  # type: ignore
    lm_mode:                    lib_sft_constants.LMModes,
    outlines_context:           "bin_sft.OutlinesContextABC",
    output_type:                lib_sft_constants.OutputTypes,
    qty_eval_small:             int,
    tok_max_query_length:       int,
    tok_max_total_length:       int,
    train_batch_size:           int,
    seed:                       int,
    subset_data:                str,
    use_workers:                bool,
):
    ###########################################################################
    # Datasets
    ###########################################################################
    assert forward_tokenizer.pad_token == prediction_tokenizer.pad_token
    assert forward_tokenizer.eos_token == prediction_tokenizer.eos_token
    small_eval_set = "small_eval"

    ###########################################################################
    # Collator
    ###########################################################################
    assert forward_tokenizer is not prediction_tokenizer
    if (
        lm_mode == lib_sft_constants.LMModes.CAUSAL_FULL and 
        output_type != lib_sft_constants.OutputTypes.OUTLINES
    ):
        if dataset_choice == lib_utils.Datasets.ARITHMETIC:
            data_collator = lib_sft_collators.ArithmeticCausalMaskedCollator(
                output_type          = output_type,
                forward_tokenizer    = forward_tokenizer,
                prediction_tokenizer = prediction_tokenizer,
                has_choices          = False
            )

        elif dataset_choice == lib_utils.Datasets.GSM8K:
            data_collator = lib_sft_collators.GSM8KCollator(
                output_type          = output_type,
                forward_tokenizer    = forward_tokenizer,
                prediction_tokenizer = prediction_tokenizer,
            )

        elif dataset_choice == lib_utils.Datasets.MATH:
            data_collator = math_sft.MATHCollator(
                forward_tokenizer    = forward_tokenizer,
                prediction_tokenizer = prediction_tokenizer,
                
            )

        else:
            raise NotImplementedError(dataset_choice)
    else:
        class CollatorChainer:
            def __init__(self, collators):
                self._collators = collators

            def __call__(self, batch):
                for collator in self._collators:
                    batch = collator(batch)
                return batch
            
        data_collator = CollatorChainer(
            (_unzip_collator, outlines_context.data_collator,)
        )

    ###########################################################################
    # Dataloaders
    ###########################################################################
    
    if dataset_choice == lib_utils.Datasets.GSM8K:
        full_ds = datasets.load_dataset("gsm8k", "main")["train"]
        gsm8k_splits = full_ds.train_test_split(test_size=0.05, seed=seed, shuffle=True)
        gsm8k_splits = {
            lib_utils.CVSets.TRAIN: gsm8k_splits["train"], 
            lib_utils.CVSets.VALID: gsm8k_splits["test" ],
            small_eval_set:         gsm8k_splits["test" ],
        }


    dataloaders = {}
    for cv_set in it.chain(lib_utils.CVSets, [small_eval_set]):
        if dataset_choice == lib_utils.Datasets.ARITHMETIC:
            ds_builder = lib_arithmetic.Arithmetic(
                    answer_only               = answer_only,
                    dataset_root_folder_dir   = data_directory,
                    eos_token                 = forward_tokenizer.eos_token,
                    extractor_ignore_one_line = extractor_ignore_one_line,
                    pad_token                 = forward_tokenizer.pad_token,
                    split                     = lib_utils.CVSets.VALID if cv_set == small_eval_set else cv_set,
                    shuffle_once              = cv_set != lib_utils.CVSets.TRAIN,
                    
                    sft_mode                  = True,
                    use_few_shots             = output_type == lib_sft_constants.OutputTypes.OUTLINES,
                    use_curriculum            = False,
                    use_cached_dataset        = True,
                    return_idx                = False,
                )
            dataset = ds_builder.make_dataset(
                difficulty_toggles=None, 
                seed=seed,
            )
            
        elif dataset_choice == lib_utils.Datasets.GSM8K:
            dataset = lib_gsm8k.GSM8K(
                tok_max_query_length  = tok_max_query_length,
                tok_max_answer_length = None,
                tok_max_total_length  = tok_max_total_length,
                ds                    = gsm8k_splits[cv_set],

                any_tokenizer         = forward_tokenizer,
                device                = LOCAL_RANK,

                use_few_shots         = False,
                few_show_qty          = None,
                
                cv_set                = cv_set,
                use_curriculum        = False,
            )

        elif dataset_choice == lib_utils.Datasets.MATH:
            dataset = math_sft.MATHDataset(
                cv_set,
                max_num_tokens = tok_max_total_length,
                forward_tokenizer = forward_tokenizer,
                prediction_tokenizer = prediction_tokenizer,
                output_type = output_type,
            )

        else:
            assert False
            ds_builder = lib_sft_commonsense_qa.openai_commonsense_qa_output(
                root_path=data_directory, 
                filter_bads=filter_bads,
            )
        
        try:
            cv_set = lib_utils.CVSets(cv_set)
        except ValueError:
            assert cv_set == small_eval_set

        if cv_set == lib_utils.CVSets.TRAIN:
            batch_size = train_batch_size
        else: 
            assert cv_set == lib_utils.CVSets.VALID or cv_set == small_eval_set, cv_set
            batch_size = eval_batch_size

        if cv_set == small_eval_set :
            dataset =  torch.utils.data.Subset(
                dataset=dataset,
                indices=range(qty_eval_small),
            )

        if subset_data:
            dataset = torch.utils.data.Subset(
                dataset=dataset,
                indices=range(int(subset_data))
            )

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            rank         = RANK,
            num_replicas = WORLD_SIZE, 
            shuffle      = cv_set == lib_utils.CVSets.TRAIN,
        )

        dataloaders[cv_set] = torch.utils.data.DataLoader(
            dataset,
            batch_size  = batch_size,
            collate_fn  = data_collator,
            sampler     = sampler,
            num_workers = get_num_cpus() if use_workers else 0,
            prefetch_factor = 10 if use_workers else None, # 2 is the default
        )

    small_dl = dataloaders[small_eval_set]
    del dataloaders[small_eval_set]
    
    return dataloaders, small_dl