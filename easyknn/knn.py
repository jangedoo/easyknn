from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Literal, Type

from easyknn.item_store import (
    InMemoryItemDataReader,
    ItemDataReader,
    ItemDataWriter,
    MemoryMappedItemDataReader,
)

logger = logging.getLogger(__name__)


class KNNBackend:
    def build(
        self, vectors, init_kwargs: dict | None = None, fit_kwargs: dict | None = None
    ):
        raise NotImplementedError()

    def get_nns_by_vector(self, vector, n: int, include_distances: bool = True):
        raise NotImplementedError()

    def get_nns_by_index(self, i: int, n: int, include_distances: bool = True):
        raise NotImplementedError()

    def save(self, path: Path | str):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: Path | str):
        raise NotImplementedError()


class SKLearnKNNBackend(KNNBackend):
    def build(
        self, vectors, init_kwargs: dict | None = None, fit_kwargs: dict | None = None
    ):
        from sklearn.neighbors import NearestNeighbors

        init_kwargs = init_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        self.index = NearestNeighbors(**init_kwargs)
        self.index.fit(vectors)
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs

    def get_nns_by_index(self, i: int, n: int, include_distances=True):
        vector = self.index._fit_X[i]
        return self.get_nns_by_vector(
            vector=vector, n=n, include_distances=include_distances
        )

    def get_nns_by_vector(self, vector, n: int, include_distances=True):
        import numpy as np

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")

        vector = vector.reshape(1, -1)
        dist_list, nbors_list = self.index.kneighbors(
            vector, n_neighbors=n, return_distance=True
        )
        if include_distances:
            return nbors_list[0], dist_list[0]
        else:
            return nbors_list[0]

    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        index_path = path / "index.pkl"
        dump = {
            "index": self.index,
            "init_kwargs": self.init_kwargs,
            "fit_kwargs": self.fit_kwargs,
        }
        pickle.dump(dump, index_path.open("wb"))

    @classmethod
    def load(cls, path: Path | str):
        path = Path(path)
        index_path = path / "index.pkl"
        dump = pickle.load(index_path.open("rb"))
        obj = cls()
        obj.fit_kwargs = dump["fit_kwargs"]
        obj.init_kwargs = dump["init_kwargs"]
        obj.index = dump["index"]
        return obj


class AnnoyBackend(KNNBackend):
    def build(
        self, vectors, init_kwargs: dict | None = None, fit_kwargs: dict | None = None
    ):
        import annoy

        init_kwargs = init_kwargs or {"metric": "euclidean"}
        fit_kwargs = fit_kwargs or {"n_trees": 10}

        ndims = len(vectors[0])
        init_kwargs["f"] = ndims

        index = annoy.AnnoyIndex(**init_kwargs)  # type: ignore
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)
        index.build(**fit_kwargs)

        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.index = index

    def get_nns_by_index(self, i: int, n: int, include_distances: bool = True):
        return self.index.get_nns_by_item(i=i, n=n, include_distances=include_distances)  # type: ignore

    def get_nns_by_vector(self, vector, n: int, include_distances: bool = True):
        return self.index.get_nns_by_vector(vector=vector, n=n, include_distances=include_distances)  # type: ignore

    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        index_path = path / "annoy.index"
        init_params_path = path / "init.pkl"
        fit_params_path = path / "fit.pkl"
        pickle.dump(self.init_kwargs, init_params_path.open("wb"))
        pickle.dump(self.fit_kwargs, fit_params_path.open("wb"))
        self.index.save(str(index_path))

    @classmethod
    def load(cls, path: Path):
        import annoy

        path = Path(path)
        index_path = path / "annoy.index"
        init_params_path = path / "init.pkl"
        fit_params_path = path / "fit.pkl"

        init_params = pickle.load(init_params_path.open("rb"))
        fit_params = pickle.load(fit_params_path.open("rb"))
        index = annoy.AnnoyIndex(**init_params)
        index.load(str(index_path))

        obj = cls()
        obj.index = index
        obj.init_kwargs = init_params
        obj.fit_kwargs = fit_params
        return obj


class FAISSBackend(KNNBackend):
    def build(
        self, vectors, init_kwargs: dict | None = None, fit_kwargs: dict | None = None
    ):
        import faiss
        import numpy as np

        fit_kwargs = fit_kwargs or {}
        init_kwargs = init_kwargs or {}
        is_binary_index = init_kwargs.get("is_binary_index", False)
        factory_fn = (
            faiss.index_binary_factory if is_binary_index else faiss.index_factory
        )
        factory_str = init_kwargs.get(
            "factory_str", "BFlat" if is_binary_index else "Flat"
        )

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        ndims = len(vectors[0]) * (8 if is_binary_index else 1)

        self.index = factory_fn(ndims, factory_str)

        if not self.index.is_trained:
            self.index.train(vectors)

        self.index.add(vectors)

        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs

    @property
    def is_binary_index(self):
        return self.init_kwargs.get("is_binary_index", False)

    def get_nns_by_index(self, i: int, n: int, include_distances: bool = True):
        vector = self.index.reconstruct(i)
        return self.get_nns_by_vector(vector, n=n, include_distances=include_distances)

    def get_nns_by_vector(self, vector, n: int, include_distances: bool = True):
        import numpy as np

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")

        vector = vector.reshape(1, -1)

        dist_list, nbors_list = self.index.search(vector, n)
        if include_distances:
            return nbors_list[0], dist_list[0]
        else:
            return nbors_list[0]

    def save(self, path: Path | str):
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        index_path = path / "faiss.index"
        init_params_path = path / "init.pkl"
        fit_params_path = path / "fit.pkl"
        pickle.dump(self.init_kwargs, init_params_path.open("wb"))
        pickle.dump(self.fit_kwargs, fit_params_path.open("wb"))

        if self.is_binary_index:
            faiss.write_index_binary(self.index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, path: Path | str):
        import faiss

        path = Path(path)
        index_path = path / "faiss.index"
        init_params_path = path / "init.pkl"
        fit_params_path = path / "fit.pkl"

        init_params = pickle.load(init_params_path.open("rb"))
        fit_params = pickle.load(fit_params_path.open("rb"))

        is_binary_index = init_params.get("is_binary_index", False)
        if is_binary_index:
            index = faiss.read_index_binary(str(index_path))
        else:
            index = faiss.read_index(str(index_path))

        obj = cls()
        obj.index = index
        obj.init_kwargs = init_params
        obj.fit_kwargs = fit_params
        return obj


class EmbeddingsIndexBuilder:
    def __init__(self):
        self.embeddings = []
        self.item_key2idx = {}
        self.items = []

    def add(self, embeddings, item_keys, items=None):
        assert len(embeddings) == len(
            item_keys
        ), "Length of embeddings and items must be same"
        if items is not None:
            assert len(embeddings) == len(
                items
            ), "Length of embeddings and items must be same"

        if len(self.embeddings) > 0 and len(self.items) == 0 and items:
            raise ValueError(
                "Previously the items were not added, so it cannot be added now"
            )

        if len(self.items) > 0 and not items:
            raise ValueError(
                "Items were added previously so they have to be added in subsequent calls as well"
            )
        for item_idx, (embedding, item_key) in enumerate(zip(embeddings, item_keys)):
            existing_idx = self.item_key2idx.get(item_key)
            item = items[item_idx] if items else None
            if existing_idx is not None:
                self.embeddings[existing_idx] = embedding
                if item:
                    self.items[existing_idx] = item
            else:
                self.embeddings.append(embedding)
                if item:
                    self.items.append(item)
                self.item_key2idx[item_key] = len(self.embeddings) - 1

    def build(
        self,
        init_kwargs: dict,
        fit_kwargs: dict,
        backend_cls: Type[KNNBackend] = AnnoyBackend,
    ) -> KNNBackend:
        backend = backend_cls()
        backend.build(self.embeddings, init_kwargs=init_kwargs, fit_kwargs=fit_kwargs)
        return backend


class EasyKNN:
    def __init__(
        self,
        index: KNNBackend,
        item_key2idx: dict,
        item_data_reader: ItemDataReader | None = None,
    ):
        self.index = index
        self.item_key2idx = item_key2idx
        self.idx2item_key = {idx: item for item, idx in self.item_key2idx.items()}
        self.item_data_reader = item_data_reader

    def resolve_items(self, nbor_idxs):
        if self.item_data_reader is not None:
            return [self.item_data_reader.query_by_idx(i) for i in nbor_idxs]
        return [self.idx2item_key[i] for i in nbor_idxs]

    def neighbors(self, vector, k=10):
        nbor_idxs, distances = self.index.get_nns_by_vector(
            vector=vector, n=k, include_distances=True
        )
        return self.resolve_items(nbor_idxs=nbor_idxs), distances

    def neighbors_by_item(self, item, k=10):
        item_idx = self.item_key2idx[item]
        return self.neighbors_by_index(item_idx, k=k)

    def neighbors_by_index(self, idx, k=10):
        nbor_idxs, distances = self.index.get_nns_by_index(
            i=idx, n=k, include_distances=True
        )
        return self.resolve_items(nbor_idxs=nbor_idxs), distances

    @classmethod
    def load(cls, path: Path | str):
        path = Path(path)
        meta_path = path / "easyknn.pkl"
        item_data_path = path / "item_data.pkl"
        index_path = path
        dump = pickle.load(meta_path.open("rb"))
        backend_cls_name = dump["backend_cls"]
        if backend_cls_name == SKLearnKNNBackend.__name__:
            index = SKLearnKNNBackend.load(index_path)
        elif backend_cls_name == AnnoyBackend.__name__:
            index = AnnoyBackend.load(index_path)
        elif backend_cls_name == FAISSBackend.__name__:
            index = FAISSBackend.load(index_path)
        else:
            raise Exception(
                f"Unsupported backend [{backend_cls_name}] specified. Cannot load."
            )

        item_key2idx = dump["item_key2idx"]
        obj = cls(
            index=index,
            item_key2idx=item_key2idx,
            item_data_reader=MemoryMappedItemDataReader(item_data_path),
        )
        return obj

    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta_path = path / "easyknn.pkl"
        item_data_path = path / "item_data.pkl"
        index_path = path

        self.index.save(path=index_path)
        pickle.dump(
            {
                "backend_cls": self.index.__class__.__name__,
                "item_key2idx": self.item_key2idx,
            },
            meta_path.open("wb"),
        )

        if isinstance(self.item_data_reader, InMemoryItemDataReader):
            ItemDataWriter(item_data_path).write(self.item_data_reader.items)
        elif isinstance(self.item_data_reader, MemoryMappedItemDataReader):
            # copy the data file and meta file to the currently specified place
            shutil.copy2(self.item_data_reader.file_path, item_data_path)
            shutil.copy2(self.item_data_reader.meta_path, path / "item_data.meta")
        else:
            logger.warning("Item metadata has not been saved.")

    @classmethod
    def from_builder(
        cls,
        builder: EmbeddingsIndexBuilder,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        backend: Literal["annoy", "sklearn", "faiss"] = "annoy",
    ):
        init_kwargs = init_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        if backend == "annoy":
            backend_cls = AnnoyBackend
        elif backend == "sklearn":
            backend_cls = SKLearnKNNBackend
        elif backend == "faiss":
            backend_cls = FAISSBackend
        else:
            raise ValueError(
                f"Unsupported backed {backend}. Must be one of [annoy, sklearn, faiss]"
            )

        index = builder.build(
            init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, backend_cls=backend_cls
        )
        return cls(
            index=index,
            item_key2idx=builder.item_key2idx,
            item_data_reader=(
                InMemoryItemDataReader(builder.items) if builder.items else None
            ),
        )

    @classmethod
    def from_builder_with_sklearn(
        cls,
        builder: EmbeddingsIndexBuilder,
        n_neighbors: int = 5,
        metric: str = "cosine",
        radius: float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        init_kwargs: dict | None = None,
    ):
        init_kwargs = (init_kwargs or {}).update(
            {
                "n_neighbors": n_neighbors,
                "metric": metric,
                "radius": radius,
                "algorithm": algorithm,
            }
        )
        return cls.from_builder(
            builder=builder, init_kwargs=init_kwargs, fit_kwargs=None, backend="sklearn"
        )

    @classmethod
    def from_builder_with_annoy(
        cls,
        builder: EmbeddingsIndexBuilder,
        metric: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
        n_trees: int = 10,
        n_jobs: int = -1,
    ):

        init_kwargs = {"metric": metric}
        fit_kwargs = {"n_trees": n_trees, "n_jobs": n_jobs}

        return cls.from_builder(
            builder=builder,
            init_kwargs=init_kwargs,
            fit_kwargs=fit_kwargs,
            backend="annoy",
        )

    @classmethod
    def from_builder_with_faiss(
        cls,
        builder: EmbeddingsIndexBuilder,
        is_binary_index=False,
        index_factory_str: str | None = None,
    ):
        init_kwargs = {
            "is_binary_index": is_binary_index,
            "factory_str": (
                index_factory_str or ("BFlat" if is_binary_index else "Flat")
            ),
        }

        return cls.from_builder(
            builder=builder, init_kwargs=init_kwargs, fit_kwargs=None, backend="faiss"
        )
