Usage
=====

The `Kiez` class enables the usage of different nearest neighbor (NN) algorithms and different hubness reduction techniques. There are several ways to tell kiez what you want to use:

.. code-block:: python

    from kiez import Kiez

    # via string and arguments as dict
    k_inst = Kiez(
        algorithm="HNSW",
        algorithm_kwargs={"n_candidates": 10},
        hubness="LocalScaling",
        hubness_kwargs={"method": "NICDM"},
    )
    from kiez.hubness import LocalScaling

    # via class and arguments as dict
    from kiez.neighbors import HNSW

    k_inst = Kiez(
        algorithm=HNSW,
        algorithm_kwargs={"n_candidates": 10},
        hubness=LocalScaling,
        hubness_kwargs={"method": "NICDM"},
    )

    # via initialized object
    hr = LocalScaling(method="NICDM")
    nn_algo = HNSW(n_candidates=10)
    k_inst = Kiez(algorithm=nn_algo, hubness=hr)

    # You can also initalize Kiez via a json file

    # content of 'conf.json' file
    # {
    #   "algorithm": "HNSW",
    #   "algorithm_kwargs": {
    #     "n_candidates": 10
    #   },
    #   "hubness": "LocalScaling",
    #   "hubness_kwargs": {
    #     "method": "NICDM"
    #   }
    # }

    >>> kiez = Kiez.from_path("conf.json")

With your initialized kiez instance you are ready to fit your data and retrieve the k nearest neighbors utilizing hubness reduction:

.. code-block:: python

    # create example data
    import numpy as np
    rng = np.random.RandomState(0)
    source = rng.rand(100,50)
    target = rng.rand(100,50)
    k_inst.fit(source, target)
    neigh_dist, neigh_ind = k_inst.kneighbors()

This will retrieve all nearest neighbors of source entities in the target entities.

You can also query for specific entities and a specific number of k neighbors:

.. code-block:: python

    neigh_dist, neigh_ind = k_inst.kneighbors()
    # get 2 nearest neighbors of the first 5 source entities
    k_inst.kneighbors(source_query_points=source[:5,:], k=2)

Single source case
-------------------

While the main focus of kiez is to be part of an embedding-based entity resolution process between two data sources, it can also be used to query a single data source:

.. code-block:: python

    # initialize your kiez instance as before
    # ... and then fit a single source
    k_inst.fit(source)

    # get the nearest neighbors of all source entities amongst themselves
    k_inst.kneighbors()
    # get 2 nearest neighbors of the first 5 source entities
    k_inst.kneighbors(source_query_points=source[:5,:], k=2)

Evaluation
----------

If you have gold standard matches for your entity resolution task you can calculate the hits@k:

.. code-block:: python

    from kiez.evaluate import hits
    import numpy as np
    # small example with toy nearest neighbor result
    nn_ind = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    gold = {0: 2, 1: 4, 2: 3, 3: 4}
    hits_result = hits(nn_ind, gold)
    print(hits_result)
    {1: 0.5, 5: 1.0, 10: 1.0}

The default result gives you the results for hits\@{1,5,10}.
But you can specify the ones you want:

.. code-block:: python

    hits_result = hits(nn_ind, gold,k=[5])
    print(hits_result)
    {5: 1.0}
