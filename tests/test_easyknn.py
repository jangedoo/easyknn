import tempfile

import pytest

import easyknn


@pytest.mark.parametrize("backend", ["annoy", "sklearn"])
def test_neighbors(backend: str):
    builder = easyknn.EmbeddingsIndexBuilder()
    builder.add([[1.0, 2.0], [5.0, 1.0], [1, 2.5]], items=["a", "b", "c"])

    knn = easyknn.EasyKNN.from_builder(builder=builder, backend=backend)
    items, distances = knn.neighbors(vector=[1.0, 2.0], k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2

    items, distances = knn.neighbors_by_item(item="a", k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2

    items, distances = knn.neighbors_by_index(idx=0, k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2


@pytest.mark.parametrize("backend", ["annoy", "sklearn"])
def test_save_an_load(backend: str):
    builder = easyknn.EmbeddingsIndexBuilder()
    builder.add([[1.0, 2.0], [5.0, 1.0], [1, 2.5]], items=["a", "b", "c"])

    knn = easyknn.EasyKNN.from_builder(builder=builder, backend=backend)

    with tempfile.TemporaryDirectory() as tempdir:
        knn.save(path=tempdir)
        knn2 = easyknn.EasyKNN.load(path=tempdir)

    nbors, dists = knn.neighbors_by_item(item="a", k=2)
    nbors2, dists2 = knn2.neighbors_by_item(item="a", k=2)

    assert nbors == nbors2
    assert len(dists) == len(dists2)
