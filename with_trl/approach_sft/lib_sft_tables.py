import os
import sys

import more_itertools as mi
import rich
import rich.markup
import rich.table
import torch
import transformers

import lib_base_classes

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def predict_table(
    *,
    batch: lib_base_classes.DataListContainer,
    predict_tokenizer, 
    predictions, 
    split, 
    epoch, 
    qty_print, 
    local_metric_outputs, 
    predictions_batch_obj,
    tok_predict_query,
):
    
    decoded_questions = batch.detok_ref_query
    decoded_answers = predict_tokenizer.batch_decode(predictions)

    table = rich.table.Table(
        title=f"{split} - Predictions, epoch {epoch}",
        show_lines=True,
    )

    table.add_column("Question:")
    table.add_column("Prediction:")
    table.add_column("Gen A:")
    table.add_column("Ref A:")
    table.add_column("Qty Toks:")

    for (
        decoded_question,
        decoded_answer,
        local_em_gen,
        local_em_ref,
        pred_tok,
        q_tok,
    ) in mi.take(
        qty_print,
        mi.zip_equal(
            decoded_questions,
            decoded_answers,
            local_metric_outputs["exact_match"].logging_columns["gen"],  # type: ignore
            local_metric_outputs["exact_match"].logging_columns["ref"],  # type: ignore
            predictions_batch_obj,
            tok_predict_query.input_ids,
        ),
    ):
        # We could skip this by decoding instead of using the pre-decoded.
        # It's still weird. I feel like it should be the same.
        escaped_q = rich.markup.escape(decoded_question).strip() # type: ignore
        escaped_a = rich.markup.escape(decoded_answer).strip() # type: ignore

        table.add_row(
            escaped_q,
            f"[green]{escaped_a}",
            rich.markup.escape(str(local_em_gen)),
            rich.markup.escape(str(local_em_ref)),
            str(len(pred_tok.response_tensor) - len(q_tok) - (pred_tok.response_tensor == predict_tokenizer.pad_token_id).sum().item()),            
        )

    rich.print(table)


def batch_table(
    *,
    batch: lib_base_classes.DataListContainer,
    is_forward: bool,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    print_qty: int,
    epoch_idx: int,
    num_epochs: int,
    batch_idx: int,
    num_batches: int,
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
