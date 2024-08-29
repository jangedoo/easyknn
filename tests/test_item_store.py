import tempfile
from pathlib import Path

import pytest

from easyknn.item_store import ItemDataWriter, MemoryMappedItemDataReader


@pytest.fixture
def items():
    return [{"name": f"name {i}", "age": i + 30} for i in range(100)]


def test_memory_mapped_item_data_reader(items):
    with tempfile.NamedTemporaryFile() as f:
        path = Path(f.name)
        writer = ItemDataWriter(path)
        writer.write(items)

        reader = MemoryMappedItemDataReader(path)
        assert len(reader.offset_length_pairs) == len(items)
        for i, item in enumerate(items):
            assert reader.query_by_idx(i) == item


def test_item_data_writer(items):
    with tempfile.NamedTemporaryFile() as f:
        path = Path(f.name)
        writer = ItemDataWriter(path)
        writer.write(items)

        assert path.stat().st_size > 0
        assert (path.parent / (path.stem + ".meta")).exists()
