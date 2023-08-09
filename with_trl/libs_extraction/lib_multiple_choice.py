import re
import libs_extraction.lib_base


class MultipleChoiceRegexExtractor(libs_extraction.lib_base.Extractor):
    def __init__(self, choices):
        self._choices = choices
        self._pat = re.compile(r"\w+")

        # each choice should parse to a single thing,
        # otherwise the extractor will break
        test_pat = [self._pat.findall(c) for c in choices]
        assert all(len(l) == 1 for l in test_pat), test_pat

    @classmethod
    def _rindex(cls, l, thing, default=None):
        if isinstance(l, str):
            try: 
                index = l.rindex(thing)
            except ValueError as err:
                if str(err) == "substring not found":
                    return default
                raise
            return index

        for index, item in enumerate(reversed(l)):
            if item == thing:
                return len(l) - index - 1
                
        return default

    @property
    def choices(self):
        return self._choices

    def parse_one(self, text):
        text = text.strip().split("\n")[0]
        
        text = self._pat.findall(text)
        pairs = []
        for choice in self._choices:
            pairs.append((choice, self._rindex(text, choice, -1)))
        return max(pairs, key=lambda s: s[1])[0]
    
    def parse(self, batch):
        return [self.parse_one(x) for x in batch]

    def compare(self, extracted_answer_a, extracted_answer_b):
        assert extracted_answer_a in self._choices, (
            extracted_answer_a, self._choices)

        assert extracted_answer_b in self._choices, (
            extracted_answer_b, self._choices)
        
        return extracted_answer_a == extracted_answer_b

    __call__ = parse_one



class MultipleChoiceRfindExtractor(libs_extraction.lib_base.Extractor):
    def __init__(self, choices):
        for choice in choices:
            assert isinstance(choice, str), type(choice).mro()

        self._choices = choices

    @property
    def choices(self):
        return self._choices

    def parse_one(self, text):
        indices = {}

        for choice in self._choices:
            indices[choice] = text.rfind(choice)
        
        key, index = max(indices.items(), key=lambda kv: kv[1])

        if index != -1:
            return key
        
        return None
    
    def parse(self, batch):
        return [self.parse_one(x) for x in batch]

    def compare(self, extracted_answer_a, extracted_answer_b):
        if extracted_answer_a is None or extracted_answer_b is None:
            return False

        assert extracted_answer_a in self._choices, (
            extracted_answer_a, self._choices)

        assert extracted_answer_b in self._choices, (
            extracted_answer_b, self._choices)
        
        return extracted_answer_a == extracted_answer_b

    __call__ = parse_one

if __name__ == "__main__":
    import ipdb; 
    ipdb.set_trace()