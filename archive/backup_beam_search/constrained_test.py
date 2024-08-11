#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import rich.status
import rich.rule

with rich.status.Status("Importing.", spinner="weather"):
    import more_itertools as mit
    import collections
    import gc
    import itertools
    import threading
    import time
    
    import fire
    import nvgpu
    import rich
    import rich.console
    import rich.markup
    import torch
    import transformers
    import mlc_datasets

    transformers.generation.utils
    CONSOLE = rich.console.Console(highlight=True)


def fake_text(*, tokenizer, text, len_, bs):
    encoded = tokenizer.encode(text)
    cycle = itertools.cycle(encoded)
    slice_ = list(itertools.islice(cycle, len_))
    tensor_ = torch.tensor(slice_).unsqueeze(0).repeat(bs, 1)
    CONSOLE.print(f"[red bold]tensor_.shape:[white] {tensor_.shape}")

    return dict(
        attention_mask=torch.ones_like(tensor_),
        input_ids=tensor_,
    )


def print_gpu_info(delai, event):
    while not event.is_set():
        info = nvgpu.gpu_info()[0]
        del info["index"]
        del info["uuid"]
        del info["type"]
        CONSOLE.print(info)
        time.sleep(delai)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _prepare_question(self, sample):
        question_text = sample["question"].strip()

        return f"Question: {question_text} Just give the final answer, then stop: "

    def _prepare_answer(self, sample):
        extracted_answer_text = sample["answer"].rstrip().rsplit(None, 1)[-1] + self.tokenizer.eos_token

        return extracted_answer_text

    def __call__(self, batch):
        tokenized_question = self.tokenizer(
            [self._prepare_question(sample) for sample in batch], 
            padding=True, return_tensors="pt"
        )
        
        tokenized_extracted_answer_batch = self.tokenizer(
            [self._prepare_answer (sample) 
             for sample in batch], 
             padding=False,
             add_special_tokens=False,
        )["input_ids"]

        return dict(
            question=tokenized_question,
            answer  =tokenized_extracted_answer_batch
        )


def main(
    bs=1,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",#"microsoft/phi-2",

    device=0,
    num_beams=3,

    new_text_len=300,
    low_memory_config = False,

    use_cache=True,
    precisions=["high",],

    load_dtype=torch.float32,
    attn_impl=None,
    # attn_impl="flash_attention_2",

    test_n_times=1,
):

    args = locals()
    for k, v in args.items():
        CONSOLE.print(f"\t{k}: {v}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    gsm8k = mlc_datasets.load_dataset("gsm8k", "main", split="train")

    dataloader = torch.utils.data.DataLoader(
        gsm8k,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=Collator(tokenizer=tokenizer),
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation=attn_impl,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    ).to(device)
    model = model.eval()

    dtypes_and_locations = collections.Counter(
        (param.device, param.dtype) 
        for param in model.parameters()
    )
    CONSOLE.print(f"{dtypes_and_locations = }")

    maybe_end = threading.Event()
    thread = threading.Thread(target=print_gpu_info, args=(0.2, maybe_end,))
    # thread.start()

    times = []
    try:
        for batch in itertools.islice(dataloader, test_n_times):
            for precision in precisions:
                if precision is not None:
                    torch.set_float32_matmul_precision(precision)

                print(f"FP32 Precision: {torch.get_float32_matmul_precision()}")

                gc.collect()
                torch.cuda.empty_cache()
                assert batch.keys() == {"question", "answer"}, batch.keys()

                forced_word_ids = batch["answer"]
                
                generation_kwargs = dict(
                    num_beams            = num_beams,
                    num_return_sequences = num_beams,

                    do_sample            = True,
                    
                    max_new_tokens       = new_text_len,
                    # min_new_tokens       = 0,
                    # min_length           = 0,

                    early_stopping       = True,
                    synced_gpus          = False,

                    low_memory           = low_memory_config,
                    use_cache            = use_cache,
                    
                    force_words_ids      = forced_word_ids,
                    
                    repetition_penalty   = 1.,
                    
                    eos_token_id         = tokenizer.eos_token_id,
                    pad_token_id         = tokenizer.eos_token_id,
                )
                start = time.perf_counter()
                tokens = model.generate(
                    **generation_kwargs,
                    **batch["question"].to(device),
                )

                # CONSOLE.print(model.config)

                print()
                CONSOLE.print(rich.rule.Rule(style="bold red"))
                forced_text = str(tokenizer.batch_decode(forced_word_ids, ignore_special_tokens=False))
                CONSOLE.print(
                    f"[red bold]Forced text:[white]   {forced_text}\n"
                    f"[red bold]Forced tokens:[white] {forced_word_ids}\n"
                    f"[red bold]EOS Token ID:[white]  {tokenizer.eos_token_id}\n"
                    f"[red bold]EOS Token:[white]     {tokenizer.eos_token}\n"
                )
                CONSOLE.print(rich.rule.Rule(style="bold red"))
                print()
                
                text_question = tokenizer.batch_decode(batch["question"]["input_ids"])[0].replace("\n", " ").strip()
                              
                for output_text, output_tokens in mit.zip_equal(
                    tokenizer.batch_decode(tokens, ignore_special_tokens=False),
                    tokens,
                ):
                    CONSOLE.print(text_question + "\n")
                    CONSOLE.print(str(batch["question"]["input_ids"]) + "\n\n")

                    CONSOLE.print(output_text.replace("\n", " ").strip() + "\n\n")
                    CONSOLE.print(str(output_tokens) + "\n")
                    CONSOLE.print(len(output_tokens) - len(batch["question"]["input_ids"][0]))
                    CONSOLE.print(rich.rule.Rule())

                delta = time.perf_counter() - start
                times.append(delta)
                # CONSOLE.print(tokenizer.batch_decode(new_tokens)[0])
                CONSOLE.print(f"{delta:0.6}s")
    finally:
        maybe_end.set()
        
        if times:
            CONSOLE.print(f"Average: {sum(times) / len(times):0.6}s")

if __name__ == "__main__":
    fire.Fire(main)
