This repository contains the description of a briefer grammar for the [dpipe](https://github.com/neuro-ml/deep_pipe) config files, 
as well as a transpiller.

# Overview
## Resources definition

Think of resources as constants in a programming language.
They are declared using the `=` symbol.

```python
file_path = "/some/path"
num_folds = 5
kernel_sizes = [3 3 5]
some_dict = {
  "a": "my string"
  "b": ["some" "list" 3]
}
```

All the data types allowed in JSON are allowed in this grammar too, however the coma (`,`) symbol is optional.

## Comments

Comments begin with `//` and end with a newline character:

```python
// a comment
x = 1 // a very important constant
```

## Modules definition

Modules are the key concept in dpipe. They can be declared in two ways: by using indentation 

```python
my_dataset = dataset.isles
    data_path = "my_path"
    filename = "meta.csv"
```

or round brackets

```python
my_dataset = dataset.isles(
    data_path = "my_path"
    filename = "meta.csv"
)
```

Note, that inside round brackets as well as square and curly brackets (JSON arrays and objects) the indentation has no effect.

So, the following declaration is equivalent to the previous one.
```python
my_dataset = dataset.isles(data_path = "my_path" filename = "meta.csv")
```

As a consequence, if you need to define a module inside an array or object, you must use round brackets:

```python
datasets = [
  dataset.isles(filename = "meta.csv") 
  dataset.brats(filename = "meta.csv")
]
```

### Nesting modules

If a module takes another module as a parameter, you can simply nest their definitions:

```python
dataset = dataset_wrapper.cached
    dataset = dataset.isles
        filename = "meta.csv"     
```

### The `init` parameter

Some modules must be initialized when they are requested. 

You can specify this behaviour with the `@init` parameter:

```json
dataset = dataset.isles
    @init = true
    filename = "meta.csv"     
```

The default values is `true`.
