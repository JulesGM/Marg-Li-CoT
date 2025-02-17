import pathlib
import hashlib
import subprocess
import collections
import rich.traceback
import tqdm
import concurrent.futures
import fire
import rich
import rich.traceback
rich.traceback.install()


def py_md5(file_path):
    file_path = pathlib.Path(file_path)
    return hashlib.md5(file_path.read_bytes()).hexdigest()

def sh_md5(file_path):
    file_path = pathlib.Path(file_path)
    return subprocess.check_output(['md5sum', file_path]).decode().split()[0]


def main(
    path = "/home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output",
    glob_pattern = "**/*.*",
):
    path = pathlib.Path(path)
    safetensors = sorted(path.glob(glob_pattern))

    indices_by_folder = collections.defaultdict(list)
    by_folder = collections.defaultdict(list)

    for safetensor in safetensors:
        run_folder_name = safetensor.parent.parent.relative_to(path)
        
        # Save the folder per run
        by_folder[run_folder_name].append(safetensor)
        
        # Save the index per run
        index = int(safetensor.parent.name.split("_")[1])
        indices_by_folder[run_folder_name].append(index)

    indices_by_folder =  {k: sorted(v) for k, v in sorted(indices_by_folder.items())}
    indices_by_folder

    # Merkel tree by folder
    merkel_tree = {}
    total_str = ""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for run_folder_name in sorted(by_folder):
            merkel_tree[run_folder_name] = []

            for safetensor in sorted(by_folder[run_folder_name]):
                merkel_tree[run_folder_name].append(executor.submit(sh_md5, safetensor))

        for run_folder_name in tqdm.tqdm(sorted(by_folder)):
            list_of_futures = merkel_tree[run_folder_name]
            assert isinstance(list_of_futures, list), type(list_of_futures)
            concatenated = "".join([future.result() for future in list_of_futures])
            per_folder_hash = hashlib.md5(concatenated.encode()).hexdigest()
            print(f"{run_folder_name}: {per_folder_hash}")
            total_str += per_folder_hash

    print(f"Overall hash: {hashlib.md5(total_str.encode()).hexdigest()}")

if __name__ == "__main__":
    fire.Fire(main) 