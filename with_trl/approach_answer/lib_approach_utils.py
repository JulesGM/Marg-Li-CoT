import json
import wandb

import numpy as np
import rich
import rich.markup
import rich.table
import torch


def convert_args_for_wandb(args):
    """ 
    Make sure that all the args are essentially JSON serializable.
    ... Essentially the same as `json.reads(json.dumps(args, default=str))`
    """

    ok_collection_types = (dict, list, tuple)
    ok_indiv_types = (int, float, str, bool)
    ok_types_all = ok_collection_types + ok_indiv_types

    if isinstance(args, dict):
        for k, v in args.items():
            assert isinstance(k, str), f"Unexpected type {type(k).mro()} for the key."

            if not isinstance(v, ok_types_all):
                args[k] = str(v)
            elif isinstance(v, ok_collection_types):
                args[k] = convert_args_for_wandb(v)
            else:
                # Logically this should only happen if in ok_indiv_types, 
                # the check is unnecessary
                assert isinstance(v, ok_indiv_types), (
                    f"Unexpected type {type(v).mro()} for {k}: {v}")

    elif isinstance(args, (list, tuple)):
        is_tuple = isinstance(args, tuple)
        args = list(args)
        for i, v in enumerate(args):
            if not isinstance(v, ok_types_all):
                args[i] = str(v)
            elif isinstance(v, ok_collection_types):
                args[i] = convert_args_for_wandb(v)
            else:
                # Logically this should only happen if in ok_indiv_types, 
                # the check is unnecessary
                assert isinstance(v, ok_indiv_types), (
                    f"Unexpected type {type(v).mro()} for {i}: {v}")

        if is_tuple:
            args = tuple(args)

    else:
        raise ValueError(f"Unexpected type {type(args).mro()} for {args}")

    return args
         

def pad_logits_across_processes(accelerator, logits, fill_value):
    size_vocab = logits[0].shape[-1]
    
    padded = []
    for sequence in logits:
        len_tensor = torch.tensor(len(sequence)).to(accelerator.device)
        global_lengths = accelerator.gather(len_tensor)
        max_len_seq = max(global_lengths, key=torch.Tensor.item)

        padded.append(
            torch.cat([
                sequence,
                torch.full(
                    (max_len_seq - len(sequence), size_vocab),
                    fill_value,
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
            ], dim=0)
        )

    return padded


def pad_logits(logits, fill_value):
    clean_logits_tensor = []
    max_len_seq = max(len(seq) for seq in logits)
    size_vocab = logits[0].shape[-1]
    
    for sequence in logits:
        clean_logits_tensor.append(
            torch.cat([
                sequence,
                torch.full(
                    (max_len_seq - len(sequence), size_vocab),
                    fill_value,
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
            ], dim=0)
        )

    return torch.stack(clean_logits_tensor, dim=0)


def _pad_check_type(expected, actual):
    if expected is None:
        expected = actual
    else:
        assert expected == actual, (
            f"Fixed type was {expected}, but found a different one {actual}")
    
    return expected


# def pad(list_of_tensors, fill_value, return_tensor=None, padding_side="right"):
#     assert padding_side == "right", padding_side
#     max_length = max(len(x) for x in list_of_tensors)
#     output_list = []
#     fixed_type = None

#     for line in list_of_tensors:
#         pad_qty = max_length - len(line)

#         if isinstance(line, torch.Tensor):
#             import ipdb; ipdb.set_trace()
#             fixed_type = _pad_check_type(fixed_type, torch.Tensor)
#             line = torch.cat([
#                 line, 
#                 torch.full((pad_qty,), fill_value, dtype=line.dtype, device=line.device)
#             ])
#         elif isinstance(line, np.ndarray):
#             fixed_type = _pad_check_type(fixed_type, np.ndarray)
#             line = np.concatenate([
#                 line, 
#                 np.full((pad_qty,), fill_value, dtype=line.dtype)
#             ])
#         elif isinstance(line, list):
#             fixed_type = _pad_check_type(fixed_type, list)
#             line = line + [fill_value] * pad_qty
#         else:
#             raise ValueError(f"Unexpected type {type(line).mro()} for {line}")

#         output_list.append(line)

#     if return_tensor == "pt" and not fixed_type == torch.Tensor:
#         if fixed_type == list:
#             return torch.tensor(output_list)
#         elif fixed_type == np.ndarray:
#             return torch.from_numpy(output_list)
#     elif return_tensor == "np" and not fixed_type == np.ndarray:
#         if fixed_type == list:
#             return np.array(output_list)
#         elif fixed_type == torch.Tensor:
#             return output_list.numpy()
#     elif (not return_tensor or return_tensor == "list") and not fixed_type == list:
#         return output_list.tolist()

def write_args_to_file(*, config_dict, output_path):
    assert "wandb_url" not in config_dict
    config_dict["wandb_url"] =  wandb.run.get_url()
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)


def print_dict_as_table(d, **table_kwargs):
    table = rich.table.Table("Key", "Value", **table_kwargs)
    for k, v in d.items():
        table.add_row(
            rich.markup.escape(str(k)), 
            rich.markup.escape(str(v)),
        )
    rich.print(table)