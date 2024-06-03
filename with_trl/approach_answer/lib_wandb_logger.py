
import more_itertools as mit
import rich
import rich.table
import wandb
import lib_constant


class WandbLoggingState:
    def __init__(self, any_tokenizer, split, also_print):
        assert False
        
        self._any_tokenizer = any_tokenizer
        self._table = wandb.Table(columns=["batch_idx", "ref_inputs", "clean_output", "raw_output"])
        self._table_just_clean_gen = wandb.Table(columns=["clean_output"])
        self._split = split
        self._also_print = also_print
        
    def log(
            self,
            *,
            batch_idx,
            batch,
            raw_output_tokens,
            clean_output_tokens,
        ):
        
        raw_output_text   = self._any_tokenizer.batch_decode(
            raw_output_tokens,   skip_special_tokens=False)
        clean_output_text = self._any_tokenizer.batch_decode(
            clean_output_tokens, skip_special_tokens=True)
        
        if self._also_print:
            self._rich_table = rich.table.Table(
                "batch_idx", 
                "ref_inputs", 
                "clean_output", 
                show_lines=True,
            )

        for clean_text, raw_text, sample in mit.zip_equal(
            clean_output_text, raw_output_text, batch,
        ):
            ref_sp = sample["ref_fs_scratchpad_gen_query_detok_not_skip"]
            end_of_few_shots = ref_sp.rfind("Q:")
            sample_specific = ref_sp[end_of_few_shots:]

            self._table.add_data(
                batch_idx,
                sample_specific,
                clean_text,
                raw_output_text
            )
            self._table_just_clean_gen.add_data(clean_text)

            if self._also_print:
                self._rich_table.add_row(
                    str(batch_idx),
                    rich.markup.escape(sample_specific),
                    rich.markup.escape(clean_text),
                    # rich.markup.escape(raw_text),
                )

        rich.print(self._rich_table)
        
        try:
            shared_header = f"{lib_constant.WANDB_NAMESPACE}_{self._split}"
            wandb.log({f"{shared_header}/batch": self._table})
            wandb.log({f"{shared_header}/just_clean_gen": self._table_just_clean_gen})

        except Exception as e:
            rich.print(f"[red bold]{e}")
            rich.print("Wandb logging failed. Continuing without logging.")

    __call__ = log