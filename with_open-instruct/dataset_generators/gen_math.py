"""
Similar to create_math_dataset, but in script form, and more general

"""
import datasets
import more_itertools as mit
import sys
sys.path.append("../open-instruct")
from eval.MATH.utilities import last_boxed_only_string, remove_boxed
from eval.MATH.minerva_utils import normalize_final_answer, get_unnormalized_answer, is_equiv
import fire
import rich
import rich.panel
import rich.markup
import transformers
import tqdm


def chat_box(role, role_input, role_style):
    role_input = rich.markup.escape(role_input)

    panel = rich.panel.Panel(
        role_input, 
        title=f"{role_style}{role}[/]:", 
        title_align="left",
        highlight=True,
    )
    
    rich.print(panel)

def normalize(solution):
    return normalize_final_answer(remove_boxed(last_boxed_only_string((solution))))


def extract_few_shot(messages):
    few_shot, question = mit.one(messages)["content"].rsplit("\n\n", 1)
    few_shot_qa = [x for x in few_shot.split("Question:") if x]
    few_shot_messages = []
    for i, qa in enumerate(few_shot_qa):
        question, answer = qa.split("Answer:")
        question = question.strip()
        answer = answer.strip()
        few_shot_messages.append({
            "content": question,
            "role": "user"
        })
        few_shot_messages.append({
            "content": answer,
            "role": "assistant"
        })

    return few_shot_messages


def main(
        fixed_few_shot=True, 
        dry=True, 
        hub_name="JulesGM/math_fixed_few_shots_with_test"
    ):

    # We go from the original dataset for improved reproducibility
    original_train = datasets.load_dataset("ai2-adapt-dev/math_ground_truth", split="train")

    # Extract the few-shot from the first message
    few_shot_messages = extract_few_shot(original_train["messages"][0])    
    other_few_shot_messages = extract_few_shot(original_train["messages"][123])
    assert few_shot_messages == other_few_shot_messages, (
        "Few-shot messages are not the same: " + 
        str(few_shot_messages) + 
        " vs " + 
        str(other_few_shot_messages)
    )

    ds = datasets.load_dataset("hendrycks/competition_math")

    if fixed_few_shot:
        set_names = ["train", "test"]
        sets = {name: [] for name in set_names}
        for set_name in set_names:
            for example in tqdm.tqdm(ds[set_name], desc=f"Processing {set_name}"):
                sets[set_name].append({
                    "messages": 
                    few_shot_messages + 
                    [{
                        "content": example["problem"].strip(), 
                        "role": "user"
                    }],
                    "ground_truth": normalize(example["solution"]),
                    "dataset": "MATH",
                })

        print("Creating final dataset")
        final = datasets.DatasetDict(
            dict(
                train=datasets.Dataset.from_list(sets["train"]),
                test=datasets.Dataset.from_list(sets["test"])
            )
        )
        print("Done")
    else:
        tests = []
        for example in ds["test"]:
            tests.append({
                "messages": [{
                    "content": few_shot.strip() + "\n\n" + example["problem"].strip(), 
                    "role": "user"
                }],
                "ground_truth": normalize(example["solution"]),
                "dataset": "MATH",
            })

        test = datasets.Dataset.from_list(tests)
        final = datasets.DatasetDict(dict(train=original_train, test=test))

    print("here")
    for name, val in final.items():
        chat_box(name, str(val["messages"][0]), "[bold blue]")
    
    print(final)


    if not dry:
        final.push_to_hub(hub_name)


if __name__ == "__main__":
    fire.Fire(main)