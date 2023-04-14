import builtins
import io
import pickle
import zipfile

import fire
import rich



class FakeClass:
    def __init__(self, *args, **kwargs):
        """ Pretend that we take any number of args.
        """

    def __setitem__(self, name, val):
        """ Pretend that we can set items.
        """

    def __call__(self, *args, **kwargs):
        """ Pretend that we can be called.
        """

    def __repr__(self):
        return "<Skipped>"


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == "builtins":
            return getattr(builtins, name) 
        return FakeClass

    def persistent_load(self, *args, **kwargs):
        return None 


def from_path_or_bytesio(path_or_bytesio):
    with zipfile.ZipFile(path_or_bytesio, "r") as zin:
        with zin.open("archive/data.pkl") as fin:
            return RestrictedUnpickler(fin).load()


def main(path):
    rich.print(from_path_or_bytesio(path))
    

if __name__ == "__main__":
    fire.Fire(main)
