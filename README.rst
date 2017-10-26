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

All the data types allowed in JSON are allowed in this grammar too,
however the coma (``,``) symbol is optional.

Comments
--------

Comments begin with ``//`` and end with a newline character:

.. code:: python

    // a comment
    x = 1 // a very important constant

Modules definition
------------------

Modules are the key concept in dpipe. They can be declared in two ways:
by using indentation

.. code:: python

    my_dataset = dataset:isles
        data_path = "my_path"
        filename = "meta.csv"

or round brackets

.. code:: python

    my_dataset = dataset:isles(
        data_path = "my_path"
        filename = "meta.csv"
    )

Note, that inside round brackets as well as square and curly brackets
(JSON arrays and objects) the indentation has no effect.

So, the following declaration is equivalent to the previous one.

.. code:: python

    my_dataset = dataset:isles(data_path = "my_path" filename = "meta.csv")

As a consequence, if you need to define a module inside an array or
object, you must use round brackets:

.. code:: python

    datasets = [
      dataset:isles(filename = "meta.csv")
      dataset:brats(filename = "meta.csv")
    ]

Nesting modules
~~~~~~~~~~~~~~~

If a module takes another module as a parameter, you can simply nest
their definitions:

.. code:: python

    dataset = dataset_wrapper:cached
        dataset = dataset:isles
            filename = "meta.csv"     

The ``lazy`` parameter
~~~~~~~~~~~~~~~~~~~~~~

Some resources must not be called when you specify their params.

To avoid the resource from being called you ca use the ``@lazy`` parameter:

.. code:: json

    dataset = dataset:isles
        @lazy
        filename = "meta.csv"     

Mixins
------

The grammar also supports multiple inheritance, realized as mixins.

.. code:: json

    @extends "../relative/path/config_one" "/or/absolute/path/config_two"

    another_resource = "Important data"

The ``@extends`` command takes any number of string arguments,
containing paths. The paths can be absolute, or relative to the folder
where lies the config that is being parsed.

Resource Manager
================

The ResourceManager class interprets a config file and manages the
resources defined in it:

.. code:: python

    rm = ResourceManager(config_path, get_module)
    print(rm.another_resource)

All the requests are processed lazily.

Registration system
===================

The RegistrationSystem class provides a convenient way to keep track of
all your modules in python code.

To register a resource (and use it in your configs) you can either use a
decorator, or a function:

.. code:: python

    @register(module_name='dummy', module_type='dataset')
    class Dataset:
        def __init__(self, data_path):
            # init implementation
            pass
        # class implementation
        pass
        
    data = 'some important data defined inside Python'
    register_inline(data, module_name='data', module_type='constants')

Then you can use it inside your config file:

.. code:: json

    dataset = dataset:dummy
        data_path = "/some/path"
        
    important = constants:data
