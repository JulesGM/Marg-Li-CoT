from pytorch_lightning.strategies import DDPStrategy as DDPS


def main():
    ddps = DDPS()
    print(ddps.global_rank)


if __name__ == "__main__":
    main()
