import abc


class Extractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text):
        pass

    @abc.abstractmethod
    def compare(self, extracted_answer_a, extracted_answer_b):
        pass