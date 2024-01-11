<p align="center">
<img src="https://github.com/dobraczka/kiez/raw/main/docs/kiezlogo.png" alt="kiez logo", width=200/>
</p>

<h2 align="center"> <a href="https://dbs.uni-leipzig.de/file/KIEZ_KEOD_2021_Obraczka_Rahm.pdf">kiez</a></h2>

<p align="center">
<a href="https://github.com/dobraczka/kiez/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/kiez/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a href='https://kiez.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/kiez/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://pypi.org/project/kiez"/><img alt="Stable python versions" src="https://img.shields.io/pypi/pyversions/kiez"></a>
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

If you have a GPU you can make kiez faster by installing [faiss](https://github.com/facebookresearch/faiss) (if you do not already have it in your environment):

``` bash
conda env create -n kiez-faiss python=3.10
conda activate kiez-faiss
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install kiez
```

For more information see their [installation instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

You can also get other specific libraries with e.g.:

``` bash
  pip install kiez[nmslib]
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
Using (A)NN libraries and hubness reduction methods:
``` python
from kiez import Kiez
import numpy as np
# create example data
rng = np.random.RandomState(0)
source = rng.rand(100,50)
target = rng.rand(100,50)
# prepare algorithm and hubness reduction
algo_kwargs = {"n_candidates": 10}
k_inst = Kiez(n_neighbors=5, algorithm="Faiss" algorithm_kwargs=algo_kwargs, hubness="CSLS")
# fit and get neighbors
k_inst.fit(source, target)
nn_dist, nn_ind = k_inst.kneighbors()
```

## Torch Support
Beginning with version 0.5.0 torch can be used, when using `Faiss` as NN library:

```python

    from kiez import Kiez
    import torch
    source = torch.randn((100,10))
    target = torch.randn((200,10))
    k_inst = Kiez(algorithm="Faiss", hubness="CSLS")
    k_inst.fit(source, target)
    nn_dist, nn_ind = k_inst.kneighbors()
```

You can also utilize tensor on the GPU:

```python

    k_inst = Kiez(algorithm="Faiss", algorithm_kwargs={"use_gpu":True}, hubness="CSLS")
    k_inst.fit(source.cuda(), target.cuda())
    nn_dist, nn_ind = k_inst.kneighbors()
```

## Documentation
You can find more documentation on [readthedocs](https://kiez.readthedocs.io)

## Benchmark
The results and configurations of our experiments can be found in a seperate [benchmarking repository](https://github.com/dobraczka/kiez-benchmarking)

## Citation
If you find this work useful you can use the following citation:
```
@article{obraczka2022fast,
  title={Fast Hubness-Reduced Nearest Neighbor Search for Entity Alignment in Knowledge Graphs},
  author={Obraczka, Daniel and Rahm, Erhard},
  journal={SN Computer Science},
  volume={3},
  number={6},
  pages={1--19},
  year={2022},
  publisher={Springer},
  url={https://link.springer.com/article/10.1007/s42979-022-01417-1},
  doi={10.1007/s42979-022-01417-1},
}
```

## Contributing
PRs and enhancement ideas are always welcome. If you want to build kiez locally use:
```bash
git clone git@github.com:dobraczka/kiez.git
cd kiez
poetry install
```
To run the tests (given you are in the kiez folder):
```bash
poetry run pytest tests
```

Or install [nox](https://github.com/theacodes/nox) and run:
```
nox
```
which checks all the linting as well.

## License
`kiez` is licensed under the terms of the BSD-3-Clause [license](LICENSE.txt).
Several files were modified from [`scikit-hubness`](https://github.com/VarIr/scikit-hubness),
distributed under the same [license](external/SCIKIT_HUBNESS_LICENSE.txt).
The respective files contain the following tag instead of the full license text.

        SPDX-License-Identifier: BSD-3-Clause

This enables machine processing of license information based on the SPDX
License Identifiers that are here available: https://spdx.org/licenses/
