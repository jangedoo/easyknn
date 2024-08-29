import pytest

from easyknn import EmbeddingsIndexBuilder


def test_add():
    builder = EmbeddingsIndexBuilder()
    # add one item embeddings
    builder.add([[1.0, 2.0]], ["a"], ["data for a"])
    # add multiple embeddings
    builder.add(
        [
            [3.0, 4.0],
            [1.0, 2.0],
        ],
        item_keys=["b", "c"],
        items=["data for b", "data for c"],
    )

    assert len(builder.embeddings) == 3
    assert builder.item_key2idx == {"a": 0, "b": 1, "c": 2}
    assert builder.items == ["data for a", "data for b", "data for c"]


def test_adding_same_item_replaces_embeddings():
    builder = EmbeddingsIndexBuilder()
    builder.add(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        item_keys=["a", "b", "c"],
        items=["a data", "b data", "c data"],
    )

    builder.add([[10, 20, 30]], item_keys=["a"], items=["new a data"])

    # check that there are three embeddings
    assert len(builder.embeddings) == 3
    # check that the first embedding is replaced with new value
    assert builder.embeddings[0] == [10, 20, 30]

    # check that the index of item a is still 0
    assert builder.item_key2idx["a"] == 0
    assert builder.items[0] == "new a data"


def test_builder_without_items():
    builder = EmbeddingsIndexBuilder()
    builder.add(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        item_keys=["a", "b", "c"],
    )

    assert len(builder.embeddings) == 3
    assert len(builder.item_key2idx) == 3
    assert len(builder.items) == 0


def test_builder_fails_with_adding_items_later():
    builder = EmbeddingsIndexBuilder()
    # add one item embeddings without item data
    builder.add([[1.0, 2.0]], ["a"])

    with pytest.raises(ValueError):
        # add multiple embeddings with item data
        builder.add(
            [
                [3.0, 4.0],
                [1.0, 2.0],
            ],
            item_keys=["b", "c"],
            items=["data for b", "data for c"],
        )


def test_builder_fails_with_not_adding_items_later():
    builder = EmbeddingsIndexBuilder()
    # add one item embeddings with item data
    builder.add([[1.0, 2.0]], ["a"], ["data for a"])

    with pytest.raises(ValueError):
        # add multiple embeddings without item data
        builder.add(
            [
                [3.0, 4.0],
                [1.0, 2.0],
            ],
            item_keys=["b", "c"],
        )
