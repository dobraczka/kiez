# kiez
[![CI](https://github.com/dobraczka/kiez/actions/workflows/main.yml/badge.svg?branch=develop)](https://github.com/dobraczka/kiez/actions/workflows/main.yml)
![testcoverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/dobraczka/7c57dda3b055c972a06f0f076df46196/raw/test.json)
![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)

A Python library for hubness reduced nearest neighbor search for the task of entity alignment with knowledge graph embeddings. The term kiez is a [german word](https://en.wikipedia.org/wiki/Kiez) that refers to a city neighborhood.

## Hubness Reduction
Hubness is a phenomenon that arises in high-dimensional data and describes the fact that a couple of entities are nearest neighbors (NN) of many other entities, while a lot of entities are NN to no one.
For entity alignment with knowledge graph embeddings we rely on NN search. Hubness therefore is detrimental to our matching results.
This library is intended to make hubness reduction techniques available to data integration projects that rely on knowledge graph embeddings in their alignment process.

## Installation
[TBD after deanonimization]

## Usage
``` python
>>> from kiez import NeighborhoodAlignment
# use 10 neighbor candidates
>>> algorithm_params = {"n_candidates":10}
# we want the 5 nearest neighbors
>>> align = NeighborhoodAlignment(n_neighbors=5,algorithm_params=algorithm_params)
# create some toy data for example purposes
>>> import numpy as np
>>> emb1 = np.random.rand(100,10)
>>> emb2 = np.random.rand(100,10)
# create indices
>>> align.fit(emb1,emb2)
<kiez.alignment.alignment.NeighborhoodAlignment object at 0x7f4c2081fdf0>
# get all nearest neighbors of emb1 in emb2
>>> dist, ind = align.kneighbors(return_distance=True)
# 5 nearest neighbors of emb[0]
>>> ind[0]
array([38,  8, 69, 79, 93])

```

License
-------
`kiez` is licensed under the terms of the BSD-3-Clause [license](LICENSE.txt).
Several files were modified from [`scikit-hubness`](https://github.com/VarIr/scikit-hubness),
distributed under the same [license](external/SCIKIT_HUBNESS_LICENSE.txt).
The respective files contain the following tag instead of the full license text.

        SPDX-License-Identifier: BSD-3-Clause

This enables machine processing of license information based on the SPDX
License Identifiers that are here available: https://spdx.org/licenses/
