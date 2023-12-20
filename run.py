import time

import faiss
import numpy as np

from kiez import Kiez, NewKiez
from kiez.hubness_reduction.new_base import NewNoHubnessReduction
from kiez.neighbors import Faiss

if __name__ == "__main__":
    source = np.random.rand(10000, 100)
    target = np.random.rand(15000, 100)

    print("==Kiez==")
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

    print("==NewKiez==")
    start = time.time()
    k_inst = NewKiez(
        n_neighbors=5,
        algorithm="Faiss",
        algorithm_kwargs=dict(index_key="HNSW", use_gpu=True),
    )
    k_inst.fit(source, target)
    dist, neigh3 = k_inst.kneighbors()
    end = time.time()
    print(end - start)

    print("==NewNoHubnessReduction==")
    start = time.time()
    nhr = NewNoHubnessReduction(Faiss(index_key="HNSW", use_gpu=True))
    nhr.fit(source, target)
    dist, neigh4 = nhr.kneighbors(5)
    end = time.time()
    print(end - start)

    print("==Faiss==")
    start = time.time()
    index = faiss.index_factory(100, "HNSW")
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(target)
    dist2, neigh2 = index.search(source, 5)
    end = time.time()
    print(end - start)
    print((neigh == neigh2).all())
    print(neigh)
    print(neigh2)
