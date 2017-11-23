This repository contains a grammar for config files, as well as a
parser, a registration system and a manager of resources, and is mainly
designed for the `deep\_pipe <https://github.com/neuro-ml/deep_pipe>`__
library.

Grammar overview
================

Resources definition
--------------------

Think of resources as constants in a programming language. They are
declared using the ``=`` symbol.

.. code:: python

    file_path = "/some/path"
    num_folds = 5
    kernel_sizes = [3 3 5]
    some_dict = {
      "a": "my string"
      "b": ["some" "list" 3]
    }

All python literals (strings, numbers) as well as dicts and lists are
supported, however the coma (``,``) symbol is optional.

Comments
--------

Just like in Python, comments begin with ``#`` and end with a newline
character (with one small exception, see below):

.. code:: python

    # a comment
    x = 1 # a very important constant

Importing stuff
---------------

The Python-like import statements are also supported:

.. code:: python

    from numpy import prod as product
    from numpy import sum, random
    import pandas as pd
    import tqdm
    import math.radians

Lazy initialization
~~~~~~~~~~~~~~~~~~~

Some resources must not be called when you specify their params.

To avoid the resource from being called you ca use the ``# lazy``
specifier:

.. code:: python

    from numpy.random import random

    random_triplet = random(
        # lazy
        size=3
    )

Now ``random_triplet`` is a function with the parameter ``size`` set to
3. In Python this is equivalent to the code below:

.. code:: python

    from numpy.random import random
    from functools import partial

    random_triplet = partial(random, size=3)

Mixins
------

The grammar also supports multiple inheritance, realized as mixins.
Importing other configs is similar to other import statements: you can
use a "starred import" or specify the path to the config. Both relative
and absolute paths are allowed.

.. code:: python

    from .parent import * # importing from the file "parent.config"
    import "../relative/path/config_one" "/or/absolute/path/config_two"
    from "prefix/folder" import 'more' 'configs'

    another_resource = "Important data"
