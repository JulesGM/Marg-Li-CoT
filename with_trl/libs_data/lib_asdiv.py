import pathlib
import typing
from typing import Any, Optional, Union

import torch
import torch.utils.data


class ASDiv(torch.utils.data.Dataset):
    def __init__(self, *, tokenizer, cache_path, quiet=False, url=None):
        self._ds = self._populate_ds(
            cache_path=cache_path,
            quiet=quiet,
            url=url,
        )
        self._tokenizer = tokenizer

        # Check that the keys are correct.
        for inner_item in self._ds:
            new_keys = {"question", "answer"}
            assert not any(k in inner_item for k in new_keys), new_keys - (
                new_keys & set(inner_item)
            )

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

        return data

    def _preprocess_question(self, question: str) -> str:
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

    def _get_indiv_item(self, idx: int):
        return dict(
            question=self._preprocess_question(self._ds[idx]["question"]),
            answer=self._ds[idx]["answer"],
        )

    def __getitem__(self, idx_or_slice: typing.Union[int, slice]):
        if isinstance(idx_or_slice, int):
            return self._get_indiv_item(idx_or_slice)

        elif isinstance(idx_or_slice, slice):
            return [
                self._get_indiv_item(i)
                for i in range(
                    idx_or_slice.start,
                    idx_or_slice.stop,
                    idx_or_slice.step,
                )
            ]

