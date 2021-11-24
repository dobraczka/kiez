Installation
============

The easiest way to get kiez is via pip:

.. code-block:: bash

   pip install kiez 

This will omit ANN libraries if you want them as well use:

.. code-block:: bash
  
  pip install kiez[all]

You can also get only a specific library with e.g.:

.. code-block:: bash
  
  pip install kiez[nmslib]


To build kiez from source use `poetry <https://python-poetry.org/>`_ 

.. code-block:: bash

   git clone git@github.com:dobraczka/kiez.git 
   cd kiez
   poetry install
