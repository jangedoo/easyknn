# easyknn
Easy KNN in Python. Wrapper for Annoy and Sklearn's Nearest neighbors implementation.

**Main highlight**: Allows searching by any hashable item rather than numeric index.
If you've worked with any nearest neighbors library like annoy, faiss or sklearn, you know that you have to write your own mapping between the numeric index of the vector and your domain object.

I wanted a minimal wrapper that takes care of building this mapping and also saving/loading the mapping together with the "index" itself.

## Example
```python
import easyknn
# create a builder
builder = easyknn.EmbeddingsIndexBuilder()

# add embeddings/vectors. anything hashable can be used as items
builder.add([[1.0, 2.0], [5.0, 1.0], [1, 2.5]], items=["a", "b", "c"])
# add more embeddings
builder.add([[1.0, 2.0], [1, 2.5]], items=["e", "f"])
# if you add new embedding for already existing item, then it's embeddings will be replaced
builder.add([[500, 200]], items=["e"])

backend = "annoy" # can be sklearn as well
knn = easyknn.EasyKNN.from_builder(builder=builder, backend=backend)
neighbors, distances = knn.neighbors_by_item(item="a", k=2)
# neighbors is a list of actual items not numeric indexes. nice !
assert all(nbor in ["a", "b", "c"] for nbor in neighbors)

# save this to a folder
knn.save("path/to/directory")

# load saved knn from a folder
knn2 = easyknn.EasyKNN.load("path/to/directory")
```

## Installation
Supports python >= 3.8 

`pip install easyknn`

## How to
There are two main components, `EmbeddingsIndexBuilder` and `EasyKNN`.

`EmbeddingsIndexBuilder` is used to gather all the items and their embeddings/vectors. Once that is done, the builder will be used to create an instance of `EasyKNN`.

`EasyKNN` provides methods like `neighbors_by_item`, `neighbors_by_index`. `save`. `load` etc.


