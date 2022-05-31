Using your own ...
===================

kiez is created with extensibility in mind. Therefore it is easy to incorporate your own hubness reduction methods, or wrappers for (approximate) nearest neighbor libraries. The reason for this is, that the central class of kiez takes hubness and nearest neighbor arguments as objects and uses them internally:

.. code-block:: python

    from kiez import Kiez
    k_inst = Kiez(algorithm=your_nn_algo, hubness=your_hubness_reduction)

... hubness reduction
---------------------

To implement your own hubness reduction class you have to extend :code:`kiez.hubness_reduction.base.HubnessReduction`.
Your class must then simply implement the methods :code:`fit` and :code:`transform`. The :code:`fit` method is called inside :code:`Kiez`'s own :code:`fit` method and receives the k-nearest neighbors information from target entities to source entities. The k value is determined by the :code:`n_candidates` value that is set in :code:`Kiez.algorithm`. Make sure you gather all the necessary data here, that you might need in the transform step.

The :code:`transform` method is called in :code:`Kiez.kneighbors` and receives the k nearest neighbors from source to target entities. Now is the time to apply your hubness reduction and return a distance matrix and new k nearest neighbors based on that distance.

For reference you can look at the :code:`kiez.hubness_reduction.CSLS` implementation.

... nearest neighbors algorithm
-------------------------------

If you are mising your favorite (approximate) nearest neighbor library, you can simply wrap it yourself.
In this case you have to extend :code:`kiez.neighbors.NNAlgorithm` and specifically the hidden :code:`_fit` and :code:`_kneighbors` functions, because :code:`fit` and :code:`kneighbors` already contain general checks and help avoid code duplication. It also takes care of handling e.g. which index is source or target.
Take a look at :code:`kiez.neighbors.SklearnNN` to see how easy it is!

The :code:`_fit` method is used to index the provided :code:`source` and :code:`target` arrays.
In the :code:`fit` method the :code:`_fit` method is called with both arrays and your job is then to simply index them:

.. code-block:: python

    # excerpt taken from kiez.neighbors.exact.sklearn_nearest_neighbors.py
    def _fit(self, data, is_source: bool):
        nn = NearestNeighbors(
            n_neighbors=self.n_candidates,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        nn.fit(data)
        return nn

Similarly, the :code:`_kneighbors` method simply wraps the necessary function:


.. code-block:: python

    # excerpt taken from kiez.neighbors.exact.sklearn_nearest_neighbors.py
    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        if is_self_querying:
            return index.kneighbors(
                X=None, n_neighbors=k, return_distance=return_distance
            )
        return index.kneighbors(X=query, n_neighbors=k, return_distance=return_distance)

In case :code:`source` and :code:`target` where found to be identical during :code:`fit` and no query was provided :code:`is_self_querying` will be provided as :code:`true`.
