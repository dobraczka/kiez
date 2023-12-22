.. _installation:

Installation
============

You can install kiez via pip:

.. code-block:: bash

    pip install kiez


If you have a GPU you can make kiez faster by installing `faiss <https://github.com/facebookresearch/faiss>`_ (if you do not already have it in your environment):

.. code-block:: bash

    conda env create -n kiez-faiss python=3.10
    conda activate kiez-faiss
    conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
    pip install autofaiss
    pip install kiez

For more information see their `installation instructions <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_.

You can also get other specific libraries with e.g.:

.. code-block:: bash

  pip install kiez[nmslib]

Other options to get specific libraries are ``nmslib``,``annoy``, ``ngt``. However faiss is the recommended library, which provides the most accurate and fastest results.


To build kiez from source use `poetry <https://python-poetry.org/>`_

.. code-block:: bash

   git clone git@github.com:dobraczka/kiez.git
   cd kiez
   poetry install
