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

# add embeddings/vectors. anything picklable can be used as items if desired
builder.add([[1.0, 2.0], [5.0, 1.0], [1, 2.5]], item_keys=["a", "b", "c"], items=["data for a", "data for b", "data for c"])
# add more embeddings
builder.add([[1.0, 2.0], [1, 2.5]], item_keys=["e", "f"], items=["data for e", "data for f"])
# if you add new embedding for already existing item, then it's embeddings will be replaced
builder.add([[500, 200]], item_keys=["e"], items=["data for e"])

backend = "annoy" # can be sklearn or faiss as well
knn = easyknn.EasyKNN.from_builder(builder=builder, backend=backend)
neighbors, distances = knn.neighbors_by_item(item="a", k=2)
# neighbors is a list of actual items not numeric indexes. nice !
# if items are not passed to the builder, items_keys will be returned instead
assert all(nbor in ["data for a", "data for c"] for nbor in neighbors)

# there are also two other functions to get nearest neighbors
# knn.neighbors(...) expects 1D vector
# knn.neighbors_by_index(...) expects an item index as input

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


