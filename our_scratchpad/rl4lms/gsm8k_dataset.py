import datasets
import rl4lms.data_pools.text_generation_pool as rl4lms_pool


def _clean_text(sample):
    return {
        k: v.replace("<<", "(").replace(">>", ")").strip() 
        for k, v in sample.items()
    }


def _split_answer_scratchpad(sample):
    scratchpad, answer = sample["answer"].split("####")
    return {
        "question": sample["question"].strip(), 
        "answer": answer.strip(), 
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
        dataset = _build_dataset(split)

        samples = []
        for idx, item in enumerate(dataset):
            sample = rl4lms_pool.Sample(
                id=f"{split}_{idx}",
                prompt_or_input_text=item["question"],
                references=[item["answer"]],
                meta_data={
                    "ref_scratchpad": item["scratchpad"],
                }
            )
            samples.append(sample)
        pool_instance = cls(samples)
        
        return pool_instance


if __name__ == "__main__":
    pool = ZeroShotGSM8KTextGenPool.prepare("train")
    print(pool)
    print(pool[0])
    print(pool[0].references)