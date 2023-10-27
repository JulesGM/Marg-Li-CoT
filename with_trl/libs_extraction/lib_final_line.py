from libs_extraction import lib_base

class FinalLineExtractor(lib_base.Extractor):
    def __init__(self, pad_token, ignore_one_line):
        super().__init__()
        self._pad_token = pad_token
        self._ignore_one_line = ignore_one_line

    def __call__(self, text):
        assert self._pad_token not in text, (
            "Text contains padding tokens.")
        
        if self._ignore_one_line:
            attempt = text.strip().rsplit("\n", 2)
            
            if len(attempt) >= 2:
                return attempt[-2].strip()
            else:
                return attempt[-1].strip()
        else:
            return text.strip().rsplit("\n", 1)[-1].strip()

    def compare(self, extracted_answer_a, extracted_answer_b):
        assert self._pad_token not in  extracted_answer_a, (
            "Text contains padding tokens.")
        assert self._pad_token not in  extracted_answer_b, (
            "Text contains padding tokens.")
        
        return extracted_answer_a == extracted_answer_b