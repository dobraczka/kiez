<p align="center">
<img src="https://github.com/dobraczka/kiez/raw/main/docs/kiezlogo.png" alt="kiez logo", width=200/>
</p>

<h2 align="center"> kiez</h2>

<p align="center">
<a href="https://github.com/dobraczka/kiez/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/kiez/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a><img alt="Test coverage" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/dobraczka/7c57dda3b055c972a06f0f076df46196/raw/test.json"></a>
<a href="https://github.com/dobraczka/kiez/blob/main/LICENSE"><img alt="License BSD3 - Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A Python library for hubness reduced nearest neighbor search for the task of entity alignment with knowledge graph embeddings. The term kiez is a [german word](https://en.wikipedia.org/wiki/Kiez) that refers to a city neighborhood.

## Hubness Reduction
Hubness is a phenomenon that arises in high-dimensional data and describes the fact that a couple of entities are nearest neighbors (NN) of many other entities, while a lot of entities are NN to no one.
For entity alignment with knowledge graph embeddings we rely on NN search. Hubness therefore is detrimental to our matching results.
This library is intended to make hubness reduction techniques available to data integration projects that rely on (knowledge graph) embeddings in their alignment process. Furthermore kiez incorporates several approximate nearest neighbor (ANN) libraries, to pair the speed advantage of approximate neighbor search with increased accuracy of hubness reduction.

## Installation
You can install kiez via pip:
``` bash
pip install kiez
```

## Usage
Simple nearest neighbor search for source entities in target space:
``` python
from kiez import Kiez
import numpy as np
# create example data
rng = np.random.RandomState(0)
source = rng.rand(100,50)
target = rng.rand(100,50)
# fit and get neighbors
k_inst = Kiez()
k_inst.fit(source, target)
nn_dist, nn_ind = k_inst.kneighbors()
```
Using ANN libraries and hubness reduction methods:
``` python
from kiez import Kiez
import numpy as np
# create example data
rng = np.random.RandomState(0)
source = rng.rand(100,50)
target = rng.rand(100,50)
# prepare algorithm and hubness reduction
from kiez.neighbors import HNSW
hnsw = HNSW(n_candidates=10)
from kiez.hubness_reduction import CSLS
hr = CSLS()
# fit and get neighbors
k_inst = Kiez(n_neighbors=5, algorithm=hnsw, hubness=hr)
k_inst.fit(source, target)
nn_dist, nn_ind = k_inst.kneighbors()
```

## Documentation
You can find more documentation on [readthedocs](https://kiez.readthedocs.io)

## Benchmark
The results and configurations of our experiments can be found in a seperate [benchmarking repository](https://github.com/dobraczka/kiez-benchmarking)

## License
`kiez` is licensed under the terms of the BSD-3-Clause [license](LICENSE.txt).
Several files were modified from [`scikit-hubness`](https://github.com/VarIr/scikit-hubness),
distributed under the same [license](external/SCIKIT_HUBNESS_LICENSE.txt).
The respective files contain the following tag instead of the full license text.

        SPDX-License-Identifier: BSD-3-Clause

This enables machine processing of license information based on the SPDX
License Identifiers that are here available: https://spdx.org/licenses/
