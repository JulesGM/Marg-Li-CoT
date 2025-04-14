import os
import multiprocessing
import subprocess
import rich
import rich.panel
import torch.distributed

class Stop:
    pass

class BroadcastParams:
    pass

class Actor(multiprocessing.Process):
    def __init__(self, model_path, gpu_id, input_queue, output_queue, world_size, init_method):
        self.gpu_id = gpu_id
        self.init_method = init_method
        self.world_size = world_size
        self.model_path = model_path
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gpu_id = gpu_id
        super().__init__()

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        import vllm

        self.model = vllm.LLM(self.model_path)
        import torch

        rich.print(f"[green]Process {self.gpu_id} loaded model. {torch.distributed.get_world_size()} {torch.distributed.get_rank()}")

        torch.distributed.destroy_process_group()

        rich.print(f"[green]Process {self.gpu_id} destroyed process group")

        torch.distributed.init_process_group(
            backend="nccl", 
            init_method=self.init_method,
            world_size=self.world_size, 
            rank=self.gpu_id,
        )
        rich.print(f"[green]Process {self.gpu_id} connected")


        while True:
            input_text = self.input_queue.get()
            if isinstance(input_text, Stop):
                break
            elif isinstance(input_text, BroadcastParams):
                for k, v in 
            print(f"Actor {self.gpu_id} received input: {input_text}")
            self.output_queue.put((self.gpu_id, self.model.chat([{"role": "user", "content": input_text}])))
        
    
def main():
    num_gpus = 3 # subprocess.check_output(["nvidia-smi", "-L"]).decode().count("\n")
    print(f"Detected {num_gpus} GPUs")
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    model_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    init_method = "tcp://localhost:23456"
    
    actors = [
        Actor(
            model_path=model_path, 
            gpu_id=i, 
            input_queue=input_queue, 
            output_queue=output_queue, 
            world_size=num_gpus, 
            init_method=init_method,
        ) for i in range(1, num_gpus)
    ]

    for actor in actors:
        actor.start()

    import torch
    torch.distributed.init_process_group(
        backend="nccl", 
        init_method=init_method,
        world_size=num_gpus, 
        rank=0,
    )
    rich.print(f"[green]Main process is running on GPU 0")

    input_queue.put("Hello, world! Who are you my friend?")
    worker_id, vllm_output = output_queue.get()

    rich.print(rich.panel.Panel(f"(worker {worker_id}) [blue]{vllm_output[0].outputs[0].text.strip()}"))

    for _ in actors:
        input_queue.put(Stop())

    for actor in actors:
        actor.join()

    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()