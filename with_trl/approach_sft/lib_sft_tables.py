import os
import sys

import more_itertools as mi
import rich
import rich.markup
import rich.table
import torch
import transformers
import wandb

import lib_base_classes
import lib_utils

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
    ) in mi.take(
        qty_print,
        mi.zip_equal(
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
            rich.markup.escape(escaped_p),         # "Prediction"
            rich.markup.escape(str(local_em_gen)), # "Extracted Gen A"
            rich.markup.escape(str(local_em_ref)), # "Ref A"
            str(p_len),                            # "Qty Toks"
        )

    wandb.log({
        f"{split.value}/predictions_table": wandb_and_rich_table.get_loggable_object()
        }, step=global_step)
    


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

    for batch_key, batch_value in batch.items():  # type: ignore
        if isinstance(batch_value, torch.Tensor) and (batch_value == -100).any():
            assert "label" in batch_key, batch_key

        for i, single_sentence in enumerate(mi.take(print_qty, batch_value)):
            if batch_key == "input_ids":
                single_sentence = [
                    x if x != -100 else forward_tokenizer.pad_token_id
                    for x in single_sentence
                ]

                text = forward_tokenizer.decode(
                    single_sentence, skip_special_tokens=False
                )

            else:
                text = str(single_sentence)

            table.add_row(
                f"{rich.markup.escape(batch_key)} - " 
                + f"{rich.markup.escape(str(i))}",
                rich.markup.escape(text),
            )
            

    if RANK == 0:
        rich.print(table)
