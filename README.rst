This repository contains an interpreter for config files written in a
subset of Python’s grammar. It aims to combine Python’s beautiful
syntax, lazy initialization and a much simpler import machinery.

Syntax overview
===============

Statements
----------

Only two Python statements are supported: variable definition and
import.

Variable definition
~~~~~~~~~~~~~~~~~~~

Variables are declared using the ``=`` operator. Because of lazy
initialization a variable cannot be redeclared, which is similar to the
``final`` behaviour in Java.

Here are some examples:

.. code:: python

    file_path = "/some/path"
    num_folds = 5
    kernel_sizes = [3, 3, 5]
    some_dict = {
      'a': 'my string',
      'b': ["some", "list", 3]
    }

Imports
~~~~~~~

You can import from other libraries inside config files just like in
regular Python:

.. code:: python

    from numpy import prod as product
    from numpy import sum, random
    import pandas as pd
    import tqdm
    import math.radians

You can also import from config files, located relative to the main
config:

.. code:: python

    # import from ./some_config.config and ./folder/dataset.config
    from .some_config import *
    from .folder.dataset import DataSet as D

You can also import configs relative to a shortcut:

.. code:: python

    # another.config content
    from assets.core import *

Provided, that resource manager knows the ``assets`` shortcut:

.. code:: python

    from resource_manager import read_config

    rm = read_config('another.config', {'assets': '/path/to/assets'})

Expressions
-----------

A large subset of Python’s expressions are supported:

.. code:: python

    # you can use other variables declared or imported in the config
    x = another_variable

    # lists, dictionaries
    l = [x, y, z]
    d = {'a': l}

    # literals
    literals = [True, False, None]

    # numbers
    nums = [1, -1, 3e-5, 0x11, 0b1001, 10_000]

    # strings
    s = 'some string'
    others = [b'one', u'two', r'three\n']
    multiline = """
        multiline
        string
    """

    # calling functions
    a = f(1, t=10, *z, y=2)

    # getting attributes
    value = x.shape

    # getting items
    item = d['a']

Comments
--------

You can also use comments inside you configs:

.. code:: python

    # comment
    x = 1 # another comment

There is also one special comment, that actually has syntactical
meaning. Consider the following example:

.. code:: python

    import numpy as np
    from functools import partial

    random_triplets = partial(np.random.uniform, size=3)

This is a common case, when you need to pass a callable with certain
parameters being fixed. Inside configs you can achieve the same effect
with the following syntax:

.. code:: python

    import numpy as np

    random_triplets = np.random.uniform(
        # lazy
        size=3
    )

Using the comment ``# lazy`` inside a function call compiles to a
corresponding functools.partial.
