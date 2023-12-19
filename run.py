import time

import faiss
import numpy as np

from kiez import Kiez

if __name__ == "__main__":
    source = np.random.rand(10000, 100)
    target = np.random.rand(15000, 100)
    start = time.time()
    k_inst = Kiez(
        n_neighbors=5,
        algorithm="Faiss",
        algorithm_kwargs=dict(index_key="HNSW", use_gpu=True),
    )
    k_inst.fit(source, target)
    dist, neigh = k_inst.kneighbors()
    end = time.time()
    print(end - start)
    start = time.time()
    index = faiss.index_factory(100, "HNSW")
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(target)
    dist2, neigh2 = index.search(source, 5)
    end = time.time()
    print(end - start)
    print((neigh == neigh2).all())
