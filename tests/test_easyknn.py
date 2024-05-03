import tempfile

import pytest

import easyknn


@pytest.fixture
def index_builder() -> easyknn.EmbeddingsIndexBuilder:
    builder = easyknn.EmbeddingsIndexBuilder()
    builder.add([[1.0, 2.0], [5.0, 1.0], [1, 2.5]], items=["a", "b", "c"])
    return builder


@pytest.mark.parametrize("backend", ["annoy", "sklearn", "faiss"])
def test_neighbors(backend: str, index_builder: easyknn.EmbeddingsIndexBuilder):
    knn = easyknn.EasyKNN.from_builder(builder=index_builder, backend=backend)
    items, distances = knn.neighbors(vector=[1.0, 2.0], k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2

    items, distances = knn.neighbors_by_item(item="a", k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2

    items, distances = knn.neighbors_by_index(idx=0, k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2


@pytest.mark.parametrize("backend", ["annoy", "sklearn", "faiss"])
def test_save_an_load(backend: str, index_builder: easyknn.EmbeddingsIndexBuilder):
    knn = easyknn.EasyKNN.from_builder(builder=index_builder, backend=backend)

    with tempfile.TemporaryDirectory() as tempdir:
        knn.save(path=tempdir)
        knn2 = easyknn.EasyKNN.load(path=tempdir)

    nbors, dists = knn.neighbors_by_item(item="a", k=2)
    nbors2, dists2 = knn2.neighbors_by_item(item="a", k=2)

    assert nbors == nbors2
    assert len(dists) == len(dists2)


def test_from_builder_with_sklearn(index_builder: easyknn.EmbeddingsIndexBuilder):
    knn = easyknn.EasyKNN.from_builder_with_sklearn(
        builder=index_builder, n_neighbors=2
    )
    assert isinstance(knn.index, easyknn.SKLearnKNNBackend)
    items, distances = knn.neighbors(vector=[1.0, 2.0], k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2


def test_from_builder_with_annoy(index_builder: easyknn.EmbeddingsIndexBuilder):
    knn = easyknn.EasyKNN.from_builder_with_annoy(builder=index_builder)
    assert isinstance(knn.index, easyknn.AnnoyBackend)
    items, distances = knn.neighbors(vector=[1.0, 2.0], k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2


def test_from_builder_with_faiss(index_builder: easyknn.EmbeddingsIndexBuilder):
    knn = easyknn.EasyKNN.from_builder_with_faiss(builder=index_builder)
    assert isinstance(knn.index, easyknn.FAISSBackend)
    items, distances = knn.neighbors(vector=[1.0, 2.0], k=2)
    assert items == ["a", "c"]
    assert len(distances) == 2
