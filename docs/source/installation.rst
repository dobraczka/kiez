.. _installation:

Installation
============

The easiest way to get kiez is via pip:

.. code-block:: bash

   pip install kiez 

To make kiez faster it is recommended to install `faiss <https://github.com/facebookresearch/faiss>`_ as well (if you do not already have it in your environment):


.. code-block:: bash

   pip install kiez[faiss-cpu]

or if you have a gpu:

.. code-block:: bash

   pip install kiez[faiss-gpu]

If you need specific cuda versions for faiss see their `installation instructions <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_ and install it seperately.

If you want all (A)NN libraries use:

.. code-block:: bash
  
  pip install kiez[all]

Other options to get specific libraries are ``nmslib``,``annoy``, ``ngt``. However faiss is the recommended library, which provides the most accurate and fastest results.


To build kiez from source use `poetry <https://python-poetry.org/>`_ 

.. code-block:: bash

   git clone git@github.com:dobraczka/kiez.git 
   cd kiez
   poetry install

