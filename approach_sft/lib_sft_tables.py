import os

import more_itertools as mit
import rich
import rich.markup
import rich.table
import torch
import transformers

from with_trl import lib_base_classes
from with_trl import lib_utils

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def predict_table(
    *,
    batch: lib_base_classes.DataListContainer,
    epoch: int,
    global_step: int,
    local_metric_outputs,
    predict_tokenizer: transformers.PreTrainedTokenizerBase,
    predictions,
    predictions_batch_obj,
    qty_print: int,
    split: lib_utils.CVSets,
    wandb_and_rich_table: lib_utils.WandbAndRichTable,
):
    assert RANK == 0, RANK

    decoded_questions = batch.detok_ref_query
    decoded_predictions = predict_tokenizer.batch_decode(predictions)
    assert predictions_batch_obj, predictions_batch_obj
    assert decoded_predictions, decoded_predictions
    assert decoded_questions, decoded_questions
    assert local_metric_outputs["exact_match"].logging_columns["gen"]
    assert local_metric_outputs["exact_match"].logging_columns["ref"]

    for (
        decoded_question,
        decoded_prediction,
        local_em_gen,
        local_em_ref,
        pred_tok,
    ) in mit.take(
        qty_print,
        mit.zip_equal(
            decoded_questions,
            decoded_predictions,
            local_metric_outputs["exact_match"].logging_columns["gen"],  # type: ignore
            local_metric_outputs["exact_match"].logging_columns["ref"],  # type: ignore
            predictions_batch_obj,
        ),
    ):
        # We could skip this by decoding instead of using the pre-decoded.
        # It's still weird. I feel like it should be the same.
        escaped_q = rich.markup.escape(decoded_question).strip() # type: ignore
        escaped_p = rich.markup.escape(decoded_prediction).strip() # type: ignore
        p_len = (len(pred_tok.response_tensor) 
            - (pred_tok.response_tensor == predict_tokenizer.pad_token_id).sum().item()
        )

        wandb_and_rich_table.add_row(
            str(epoch),                            # "Epoch"
            rich.markup.escape(escaped_q),         # "Question"
            rich.markup.escape(escaped_p.rstrip(predict_tokenizer.pad_token)), # "Prediction"
            rich.markup.escape(str(local_em_gen)), # "Extracted Gen A"
            rich.markup.escape(str(local_em_ref)), # "Ref A"
            str(p_len),                            # "Qty Toks"
        )

    wandb_and_rich_table.get_loggable_object()
    # wandb.log({
    #         f"{lib_constant.WANDB_NAMESPACE}/{split.value}/predictions_table": 
    #     }, 
    #     step=global_step,
    # )
    


def batch_table(
    *,
    batch: lib_base_classes.DataListContainer,
    batch_idx: int,
    epoch_idx: int,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    is_forward: bool,
    num_batches: int,
    num_epochs: int,
    print_qty: int,
    
):
    """
    Prints values of a batch in a table.
    """
    mode_str = (
        f"[E{epoch_idx}/{num_epochs} - B{batch_idx}/{num_batches}]:" + 
        " Forward"
        if is_forward
        else "Backward"
    )
    table = rich.table.Table(title=f"Batch {mode_str}", show_lines=True)
    table.add_column("Key", style="green bold")
    table.add_column("Value")

    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]

    assert len(batch) == 2, batch.keys()
    assert not (input_ids == -100).any()

    for i, (input_id_single, attention_mask_single) in enumerate(mit.take(print_qty, mit.zip_equal(input_ids, mask))):
        assert isinstance(input_ids, torch.Tensor), type(input_ids)
        unmasked_inputs = input_id_single[attention_mask_single.bool()]
        unmasked_inputs = [
            x if x != -100 else forward_tokenizer.pad_token_id
            for x in unmasked_inputs
        ]


        text = rich.markup.escape(forward_tokenizer.decode(
            unmasked_inputs, skip_special_tokens=False,
        ))

        for v in forward_tokenizer.special_tokens_map.values():
            text = text.replace(v, f"[bold blue]{v}[/]")

        table.add_row(
            f"input_ids - {i}",
            text,
        )

    for i, attention_mask_single in enumerate(mit.take(print_qty, mask)):
        text = str(attention_mask_single)

        table.add_row(
            f"attention_mask - {i}",
            rich.markup.escape(text),
        )
            

    if RANK == 0:
        rich.print(table)
