import enum
import random
import subprocess
import sys
import os
import typing

import fire
import jsonlines as jl
import more_itertools
import rich
import rich.markup
import rich.table
import rich.traceback

rich.traceback.install()

class CVSets(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


def run_cmd(ctx_name, cmd):
    header = f"[bold blue]-> {ctx_name}.run_cmd:[/] "
    rich.print(f"{header}Running command:\n\t{cmd}")
    output = subprocess.run(
        cmd, 
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    if output.returncode != 0:
        rich.print(f"{header}[red]Error running cmd:[/]\n\t{cmd}")
        rich.print(f"{header}[red]Stdout:[/]\n\t{output.stdout.decode('utf-8')}")
        rich.print(f"{header}[red]Stderr:[/]\n\t{output.stderr.decode('utf-8')}")
        sys.exit(1)

    rich.print(f"{header}Done running cmd.")


def split_num_to_digits(num):
    return list(str(num))


def integer_to_str(num, separate_digits, separator):

    if separate_digits:
        assert isinstance(separator, str), type(separator).mro()
        return separator.join(split_num_to_digits(num))
    
    return str(num)


def gen_examples(
        *,
        n_examples: int,
        digits,
        randomized_digits: bool,
        min_digit: int,
        max_digit: int,
        include_scratchpad: bool,
        separate_digits: bool,
        separator: typing.Optional[str],
    ):
    
    complete_str = []
    sample_dicts = []
    for _ in range(n_examples):
        if randomized_digits:
            digits = random.randrange(min_digit + 1, max_digit)

        first_number  = random.randrange(int(10 ** digits), int(10 ** (digits + 1)))
        second_number = random.randrange(int(10 ** digits), int(10 ** (digits + 1)))
        
        input_first = integer_to_str(
            first_number, 
            separator=separator,
            separate_digits=separate_digits, 
        )
        
        input_second = integer_to_str(
            second_number, 
            separator=separator,
            separate_digits=separate_digits, 
        )

        input_sum = f"{input_first} + {input_second}"

        resultant_str = (
            "Input:\n" +
            f"{input_sum}\n" +
            "Target:\n"
        )

        scratch_pad = None
        if include_scratchpad:
            scratch_pad = "<scratch>\n"
            carry       = 0
            running_sum = ""

            for first_digit, second_digit in reversed(list(
                more_itertools.zip_equal(
                    split_num_to_digits(first_number), 
                    split_num_to_digits(second_number),
                ))):
            
                dig_sum      = int(first_digit) + int(second_digit) + carry
                
                if separate_digits:
                    running_sum  = f"{dig_sum % 10}{separator}{running_sum}"
                else:
                    running_sum  = f"{dig_sum % 10}{running_sum}"

                maybe_prev_c = " + 1 (Previous carry)" if carry else ""
                carry        = int(dig_sum >= 10)
                scratch_pad += (
                    f"{first_digit} + {second_digit}{maybe_prev_c} , " +
                    f"{running_sum} Carry: {carry}\n"
                )

            # scratch_pad += f", {running_sum}C: {carry}\n"
            scratch_pad   += f"{carry} {running_sum}".strip() + "\n"
            scratch_pad   += "</scratch>\n"
            resultant_str += scratch_pad
            
        resultant_str += integer_to_str(
            first_number + second_number, 
            separator=separator,
            separate_digits=separate_digits, 
        )
        resultant_str += "\n\n"
        complete_str.append(resultant_str,)

        obj = dict(
            input      = input_sum, 
            answer     = integer_to_str(
                first_number + second_number,
                separator=separator,
                separate_digits=separate_digits, 
            ),
            num_digits = digits,
            scratchpad = scratch_pad,
        )

        sample_dicts.append(obj)
        
    return complete_str, sample_dicts


def main(
    gen_text = False,

    val_start           = 3,
    max_digits          = 5,
    n_samples           = 2000,
    include_scratchpad  = True,
    separate_digits     = True,
    separator           = " ",
    
    # For few-shots
    randomized_digits   = False,
    examples_per_prompt = 1, # Num. shots per pool
    fixed_examples      = False, # Randomize from pool
    context_examples    = 50, # Pool of few-shots
):

    args = locals()
    table = rich.table.Table("Keys", "Values", title="Arguments")
    for k, v in args.items():
        table.add_row(str(k), rich.markup.escape(str(v)))
    rich.print(table)

    if fixed_examples and randomized_digits:
        # Generate few_shot with randomized digits
        assert False
        few_shot_str, few_shot_list_dict = gen_examples(
            n_examples          = examples_per_prompt - 1,
            digits             = None,
            randomized_digits  = randomized_digits,
            min_digits         = 1,
            max_digits         = val_start - 1,
            include_scratchpad = include_scratchpad,
        )

    for cv_set in [CVSets.TRAIN, CVSets.VALIDATION]:
        lower_bound = (0 if cv_set == CVSets.TRAIN else val_start)
        for digits in range(lower_bound, max_digits):
            print(f"{cv_set.value} - {digits}")

            folder_name = cv_set.value
            
            if include_scratchpad:
                folder_name += "_scratch"

                # Potentially add few-shot scratchpads if not randomized digits
                if fixed_examples and not randomized_digits:
                    assert False
                    few_shot_str, few_shot_list_dict = gen_examples(
                        n_examples         = context_examples,
                        digits             = digits,
                        randomized_digits  = randomized_digits,
                        include_scratchpad = include_scratchpad,
                    )
                    
            else:
                folder_name += "_direct"
                
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
                
            complete_str = ""
            with jl.open(f"{folder_name}/{digits + 1}.jsonl", "w") as f:
                
                for _ in range(n_samples):
                    
                    n_gen_examples = 1 if fixed_examples else examples_per_prompt

                    if fixed_examples:
                        assert False
                        complete_str += "".join(random.sample(
                            few_shot_str, examples_per_prompt - 1
                        ))

                    # Main sample generation
                    sample_strs, sample_list_dicts = gen_examples(
                        n_examples         = n_gen_examples,
                        digits             = digits,
                        include_scratchpad = include_scratchpad,
                        randomized_digits  = False,
                        min_digit          = None,
                        max_digit          = None,
                        separate_digits    = separate_digits,
                        separator          = separator,
                    )
                    
                    for sample_dict in sample_list_dicts:
                        f.write(sample_dict)

                    complete_str += "".join(sample_strs)
                    complete_str += "<|endoftext|>"

            if gen_text:
                with open(f"{folder_name}/{digits + 1}.txt", "w") as f:
                    f.write(complete_str)

                    

if __name__ == "__main__":
    fire.Fire(main)
