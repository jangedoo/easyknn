from __future__ import annotations

import gzip
import mmap
import pickle
from pathlib import Path
from typing import Any


class ItemDataReader:
    def query_by_idx(self, idx):
        raise NotImplementedError()


class InMemoryItemDataReader(ItemDataReader):
    def __init__(self, items: list):
        self.items = items

    def query_by_idx(self, idx):
        return self.items[idx]


class MemoryMappedItemDataReader(ItemDataReader):
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.meta_path = self.file_path.parent / (self.file_path.stem + ".meta")

        if not self.file_path.exists():
            raise ValueError(f"{self.file_path} does not exist")
        if not self.meta_path.exists():
            raise ValueError(f"{self.meta_path} does not exist")

        with self.meta_path.open("rb") as f:
            self.offset_length_pairs = pickle.load(f)

        self.file = open(self.file_path, "rb")
        self.mmapped_file = mmap.mmap(
            self.file.fileno(), length=0, access=mmap.ACCESS_READ
        )

    def __del__(self):
        self.mmapped_file.close()
        self.file.close()

    def query_by_idx(self, idx):
        offset, length = self.offset_length_pairs[idx]
        self.mmapped_file.seek(offset)
        bytes = self.mmapped_file.read(length)
        item = pickle.loads(gzip.decompress(bytes))
        return item


class ItemDataWriter:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)

    def write(self, items: list[Any]):
        offset_length_pairs = []
        with open(self.file_path, "wb") as f:
            for item in items:
                offset = f.tell()
                item_bytes = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
                compressed_bytes = gzip.compress(item_bytes)
                length = f.write(compressed_bytes)
                offset_length_pairs.append((offset, length))

        meta_path = self.file_path.parent / (self.file_path.stem + ".meta")
        with open(meta_path, "wb") as f:
            pickle.dump(offset_length_pairs, f)
