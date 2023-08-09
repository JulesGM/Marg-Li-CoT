import more_itertools
import pathlib
import typing
from typing import Any, Optional, Union

import torch
import torch.utils.data

import wget
import xml

class ASDiv(torch.utils.data.Dataset):
    def __init__(self, *, any_tokenizer, cache_path, quiet=False, url=None):
        self._ds = self._populate_ds(
            cache_path=cache_path,
            quiet=quiet,
            url=url,
        )
        self._tokenizer = any_tokenizer

        super().__init__()

    @classmethod
    def _populate_ds(
        cls,
        cache_path,
        url=None,
        quiet=False,
    ):
        if url is None:
            url = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"

        cache_path = pathlib.Path(cache_path)
        url = url
        data = {}

        if not cache_path.exists():
            if not quiet:
                print("Downloading dataset...")
            wget.download(url, out=str(cache_path), bar=None)  # type: ignore
            if not quiet:
                print("Download complete.")

        if not quiet:
            print("Parsing dataset...")

        with cache_path.open() as fp:
            root = xml.etree.ElementTree.parse(fp).getroot()[0]  # type: ignore
            data = [
                {element.tag: element.text for element in x} | dict(x.items())
                for x in root
            ]

        if not quiet:
            print("Parsing complete.")


        # Invert
        data_keys = data[0].keys()
        assert [x.keys() == data_keys for x in data]

        inverted_data = {}
        for key in data_keys:
            inverted_data[key.lower()] = [x[key] for x in data]
            
        inverted_data["question"] = [
            f"{body} {question}" 
            for body, question in 
            more_itertools.zip_equal(inverted_data["body"], inverted_data["question"])
        ]
        
        return inverted_data
    

    def _preprocess_question(self, question: str) -> str:
        # Tok Detok.

        tokenized = self._tokenizer(
            question,
            add_special_tokens=False,
        )

        assert isinstance(
            tokenized["input_ids"], list
        ), f"{type(tokenized['input_ids']).mro() = }"
        assert isinstance(
            tokenized["input_ids"][0], int
        ), f"{type(tokenized['input_ids'][0]).mro() = }"

        return self._tokenizer.decode(
            tokenized["input_ids"],
            skip_special_tokens=True,
        ).strip()

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, *args, **kwargs):
        return dict(
            question=self._ds["question"].__getitem__(*args, **kwargs),
            answer=self._ds["answer"].__getitem__(*args, **kwargs),
        )

