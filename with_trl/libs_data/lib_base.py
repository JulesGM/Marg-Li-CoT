import abc
import libs_extraction.lib_base
import torch.utils.data

class Dataset(abc.ABC, torch.utils.data.Dataset):
    def get_extractor(self) -> libs_extraction.lib_base.Extractor:
        raise NotImplementedError()
