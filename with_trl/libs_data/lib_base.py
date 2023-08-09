import abc
import libs_extraction.lib_base
import torch.utils.data

class Dataset(abc.ABC, torch.utils.data.Dataset):
    @abc.abstractmethod
    def get_extractor(self) -> libs_extraction.lib_base.Extractor:
        raise NotImplementedError()

    @abc.abstractmethod
    def use_few_shots(self):
        raise NotImplementedError()


class FewShotMixin(abc.ABC):
    @classmethod
    @abc.abstractclassmethod
    def post_process_gen_fewshots(cls) -> str:
        raise NotImplementedError()