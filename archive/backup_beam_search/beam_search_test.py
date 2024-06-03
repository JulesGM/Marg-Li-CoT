#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import rich.status

with rich.status.Status("Importing.", spinner="weather"):
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


def main(
    bs=1,
    
    # model_name="susnato/phi-2",
    model_name="microsoft/phi-2",

    device=0,
    num_beams=15,
    query_len=500,
    new_text_len=20,
    
    # low_memory_config=dict(
    #     sub_batch_size=96,
    #     offload=False,
    # ),

    low_memory_config = False,
    use_cache=True,
    precisions=["high",],
    # load_dtype=torch.bfloat16,
    load_dtype=None,
    attn_impl=None, 
    # attn_impl="flash_attention_2",
    test_n_times=3,
):

    args = locals()
    for k, v in args.items():
        CONSOLE.print(f"\t{k}: {v}")

    CONSOLE.print(f"{torch.cuda.is_bf16_supported() = }")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation=attn_impl,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )

    model = model.to(device)
    model = torch.compile(model, backend="inductor")

    dtypes_and_locations = collections.Counter(
        (param.device, param.dtype) 
        for param in model.parameters()
    )
    CONSOLE.print(f"{dtypes_and_locations = }")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenized = fake_text(
        tokenizer = tokenizer,
        text = "Tell me what bfloat16 is: Bfloat16 is",
        len_ = query_len,
        bs = bs,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    maybe_end = threading.Event()
    
    try:
        thread = threading.Thread(target=print_gpu_info, args=(0.2, maybe_end,))
        thread.start()
        times = []

        for i in range(test_n_times):
            for precision in precisions:
                if precision is not None:
                    torch.set_float32_matmul_precision(precision)

                print(f"FP32 Precision: {torch.get_float32_matmul_precision()}")

                gc.collect()
                torch.cuda.empty_cache()

                start = time.perf_counter()
                # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                new_tokens = model.generate(
                
                    num_beams            = num_beams,
                    num_return_sequences = num_beams,
                    
                    do_sample            = False,
                    
                    max_new_tokens       = new_text_len,
                    min_new_tokens       = new_text_len,

                    pad_token_id         = tokenizer.eos_token_id,
                    repetition_penalty   = 1.,

                    low_memory           = low_memory_config,
                    use_cache            = use_cache,
                    
                    **tokenized,
                )
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
