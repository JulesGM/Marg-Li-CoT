import enum
import itertools
import zipfile

import pickletools
import fire


class States(enum.Enum):
    SEARCHING = 0
    READING = 1
    

def main(path, names, n=50, verbose=False):
    assert isinstance(names, (list, tuple, set)), type(names)
    assert all([isinstance(x, str) for x in names]), names
    names = set(names)    
    results = {}

    state = States.SEARCHING
    reading = None
    with zipfile.ZipFile(path, "r") as zip_f:
        with zip_f.open("archive/version", "r") as fin:
            version = int(fin.read().decode())
            assert version == 3, version

        with zip_f.open("archive/data.pkl", "r") as fin:
            for opcode, arg, pos in itertools.islice(pickletools.genops(fin), n):
                if verbose:
                    print(f"{opcode.name}, {arg = }, {pos = }")
    
                if state == States.SEARCHING:
                    if opcode.name == "BINUNICODE" and arg in names:
                        state = States.READING
                        reading = arg
                        names.remove(arg)

                elif state == States.READING:
                    if opcode.name == "BINPUT":
                        continue

                    results[reading] = arg
                    state = States.SEARCHING

                    if not names:
                        break
                else:
                    raise ValueError(state) 
        print(results)        
    return results

if __name__ == "__main__":
    fire.Fire(main)
