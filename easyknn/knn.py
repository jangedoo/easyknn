from __future__ import annotations

import pickle
from pathlib import Path
from typing import Type


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


class EmbeddingsIndexBuilder:
    def __init__(self):
        self.embeddings = []
        self.item2idx = {}

    def add(self, embeddings, items):
        assert len(embeddings) == len(
            items
        ), "Length of embeddings and items must be same"
        for embedding, item in zip(embeddings, items):
            existing_idx = self.item2idx.get(item)
            if existing_idx is not None:
                self.embeddings[existing_idx] = embedding
            else:
                self.embeddings.append(embedding)
                self.item2idx[item] = len(self.embeddings) - 1

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
    def __init__(self, index: KNNBackend, item2idx: dict):
        self.index = index
        self.item2idx = item2idx
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}

    def resolve_items(self, nbor_idxs):
        return [self.idx2item[i] for i in nbor_idxs]

    def neighbors(self, vector, k=10):
        nbor_idxs, distances = self.index.get_nns_by_vector(
            vector=vector, n=k, include_distances=True
        )
        return self.resolve_items(nbor_idxs=nbor_idxs), distances

    def neighbors_by_item(self, item, k=10):
        item_idx = self.item2idx[item]
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
        index_path = path
        dump = pickle.load(meta_path.open("rb"))
        backend_cls_name = dump["backend_cls"]
        if backend_cls_name == SKLearnKNNBackend.__name__:
            index = SKLearnKNNBackend.load(index_path)
        elif backend_cls_name == AnnoyBackend.__name__:
            index = AnnoyBackend.load(index_path)
        else:
            raise Exception(
                f"Unsupported backend [{backend_cls_name}] specified. Cannot load."
            )

        item2idx = dump["item2idx"]
        obj = cls(index=index, item2idx=item2idx)
        return obj

    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta_path = path / "easyknn.pkl"
        index_path = path

        self.index.save(path=index_path)
        pickle.dump(
            {
                "backend_cls": self.index.__class__.__name__,
                "item2idx": self.item2idx,
            },
            meta_path.open("wb"),
        )

    @classmethod
    def from_builder(
        cls,
        builder: EmbeddingsIndexBuilder,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        backend: str = "annoy",
    ):
        init_kwargs = init_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        backend = backend.lower().strip()
        if backend == "annoy":
            backend_cls = AnnoyBackend
        elif backend == "sklearn":
            backend_cls = SKLearnKNNBackend
        else:
            raise ValueError(
                f"Unsupported backed {backend}. Must be one of [annoy, sklearn]"
            )

        index = builder.build(
            init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, backend_cls=backend_cls
        )
        return cls(index=index, item2idx=builder.item2idx)
