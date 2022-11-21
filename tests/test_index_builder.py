from easyknn import EmbeddingsIndexBuilder


def test_add():
    builder = EmbeddingsIndexBuilder()
    # add one item embeddings
    builder.add([[1.0, 2.0]], ["a"])
    # add multiple embeddings
    builder.add(
        [
            [3.0, 4.0],
            [1.0, 2.0],
        ],
        items=["b", "c"],
    )

    assert len(builder.embeddings) == 3
    assert builder.item2idx == {"a": 0, "b": 1, "c": 2}


def test_adding_same_item_replaces_embeddings():
    builder = EmbeddingsIndexBuilder()
    builder.add([[1, 2, 3], [4, 5, 6], [7, 8, 9]], items=["a", "b", "c"])

    builder.add([[10, 20, 30]], items=["a"])

    # check that there are three embeddings
    assert len(builder.embeddings) == 3
    # check that the first embedding is replaced with new value
    assert builder.embeddings[0] == [10, 20, 30]

    # check that the index of item a is still 0
    assert builder.item2idx["a"] == 0
