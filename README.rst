This repository contains an interpreter for config files written in a
subset of Python’s grammar. It aims to combine Python’s beautiful
syntax, lazy initialization and a much simpler import machinery.

Features overview
=================

Lazy scopes
-----------

The scopes inside config files are lazy, this means that any declared value will be calculated
only when requested. This allows the user to define various memory-consuming values in a single
config.

Because the scopes are lazy, the following code is ambiguous:

.. code:: python

    # which value to use?
    x = 1
    x = 2


To avoid ambiguity, each name can be used only once in a config. As a consequence some
syntactic structures (e.g. ``for`` loops) are not supported.

Statements
----------

Only three Python statements are supported: value definitions, function definitions and imports.

Value definitions
~~~~~~~~~~~~~~~~~

Values are declared using the ``=`` operator followed by any valid Python expression.

Here are some examples:

.. code:: python

    file_path = "/some/path"
    num_folds = 5
    kernel_sizes = [3, 3, 5]
    some_value = apply_to_input(lambda x: x ** 2)
    odd_squares = [i ** 2 for i in range(10) if i % 2 == 1]


Function definitions
--------------------

You can define functions inside configs, however their local scope is also lazy, thus the same
constraints hold.

A function body consists of several value or function definitions or assertions followed by a return statement:

.. code:: python

    def normalize(x, y, z):
        length = sqrt(x ** 2 + y ** 2 + z ** 2)
        assert length > 0
        return x / length, y / length, z / length


    def adder(f):
        def wrapper(x):
            return f(x) + 1

        return wrapper


    @adder
    def f(x):
        return x


    def check_call(seq):
        assert seq, seq
        return f(seq[0])


Even though the scopes are lazy, all the assertions are always evaluated (just before the return statement).

Imports
~~~~~~~

You can import from other libraries inside config files just like in regular Python:

.. code:: python

    from numpy import prod as product
    from numpy import sum, random
    import pandas as pd
    import tqdm
    import math.radians

You can also import from config files, located relative to the main config (in this
case from ``./some_config.config``, ``./folder/dataset.config`` and ``../upper/another.config``):

.. code:: python

    from .some_config import *
    from ..upper.another import *
    from .folder.dataset import DataSet as D

Note, that you can use starred imports (e.g. ``from a import *``) only when importing from another config.
