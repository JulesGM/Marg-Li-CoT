import pytorch_lightning as pl
import fire

def main():
    s = pl.plugins.environments.SLURMEnvironment()
    lr = s.local_rank()
    gr = s.global_rank()
    nr = s.node_rank()
    print(f"[{gr} - {nr}:{lr}]")

if __name__ == "__main__":
    fire.Fire(main)
