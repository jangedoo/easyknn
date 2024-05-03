from __future__ import annotations

import tempfile

import numpy as np
import pytest

from easyknn import AnnoyBackend, FAISSBackend, KNNBackend, SKLearnKNNBackend


def test_build():
    backend = SKLearnKNNBackend()
    backend.build(vectors=[[1, 2], [4, 10], [1, 2.5]])
    assert backend.index is not None


@pytest.mark.parametrize(
    "backend", [SKLearnKNNBackend(), AnnoyBackend(), FAISSBackend()]
)
def test_get_nns_by_index(backend: KNNBackend):
    backend.build(vectors=[[1.0, 2.0], [5.0, 1.0], [1, 2.5]])

    # fetch only 2 neighbors
    nbors, dists = backend.get_nns_by_index(i=0, n=2, include_distances=True)
    # for item at 0th pos, we expect items with following index: 0 and 2 because they are closer
    assert list(nbors) == [0, 2]
    assert len(dists) == 2

    # we expect only neighbors to be returned when include distances = False
    nbors = backend.get_nns_by_index(i=0, n=2, include_distances=False)
    assert list(nbors) == [0, 2]


@pytest.mark.parametrize(
    "backend", [SKLearnKNNBackend(), AnnoyBackend(), FAISSBackend()]
)
def test_get_nns_by_vector(backend: KNNBackend):
    backend.build(vectors=[[1.0, 2.0], [5.0, 1.0], [1, 2.5]])

    # fetch only 2 neighbors
    nbors, dists = backend.get_nns_by_vector(
        vector=[1, 2.1], n=2, include_distances=True
    )
    # we expect items at 0 and 2nd index to be returned
    assert list(nbors) == [0, 2]
    assert len(dists) == 2

    # we expect only neighbors to be returned when include distances = False
    nbors = backend.get_nns_by_vector(vector=[1, 2.1], n=2, include_distances=False)
    assert list(nbors) == [0, 2]


@pytest.mark.parametrize(
    "backend,init_kwargs",
    [
        (SKLearnKNNBackend(), None),
        (AnnoyBackend(), None),
        (FAISSBackend(), {"is_binary_index": False}),
        (FAISSBackend(), {"is_binary_index": True}),
    ],
)
def test_save_and_load(backend: KNNBackend, init_kwargs):
    vectors = np.random.normal(size=(10, 128))
    if (init_kwargs or {}).get("is_binary_index", False):
        vectors = np.packbits(vectors > 0).reshape(vectors.shape[0], -1)

    backend.build(
        vectors=vectors,
        init_kwargs=init_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmppath:
        backend.save(path=tmppath)

        loaded_backend = backend.__class__.load(path=tmppath)

    assert backend.fit_kwargs == loaded_backend.fit_kwargs
    assert backend.init_kwargs == loaded_backend.init_kwargs

    # check that we still get correct results from loaded index as well
    nbors = loaded_backend.get_nns_by_index(i=0, n=2, include_distances=False)
    assert len(nbors) == 2
