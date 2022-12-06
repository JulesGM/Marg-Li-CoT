import datasets
import rich
import rl4lms.data_pools.text_generation_pool as rl4lms_pool
import rl4lms.envs.text_generation.registry as rl4lms_registry

datasets.logging.set_verbosity_error()


def _clean_text(sample):
    return {
        k: v.replace("<<", "(").replace(">>", ")").strip() 
        for k, v in sample.items()
    }


def _split_answer_scratchpad(sample):
    scratchpad, answer = sample["answer"].split("####")
    return {
        "question":   sample["question"].strip(), 
        "answer":     answer.strip(), 
        "scratchpad": scratchpad.strip()
    }


def _build_dataset(split):
    assert split in ("train", "test"), split
    dataset = datasets.load_dataset("gsm8k", "main", split=split)
    dataset = dataset.map(_clean_text).map(_split_answer_scratchpad)
    return dataset


class ZeroShotGSM8KTextGenPool(rl4lms_pool.TextGenPool):
    @classmethod
    def prepare(cls, split: str):
        if split == "val":
            split = "test"
            
        dataset = _build_dataset(split)

        samples = []
        for idx, item in enumerate(dataset):
            sample = rl4lms_pool.Sample(
                id                   = f"{split}_{idx}",
                meta_data            = {"ref_scratchpad": item["scratchpad"],},
                references           = [item["answer"]],
                prompt_or_input_text = item["question"],
            )
            samples.append(sample)
        pool_instance = cls(samples)
        
        return pool_instance


rl4lms_registry.DataPoolRegistry.add(
    "zero_shot_gsm8k_text_gen_pool",
    ZeroShotGSM8KTextGenPool,
)


if __name__ == "__main__":
    pool = ZeroShotGSM8KTextGenPool.prepare("train")
    rich.print(pool[3])