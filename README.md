Easy config files in pure Python!

What you can do directly in your configs:

- define constants
- define functions
- use import statements
- call functions and apply arithmetic operations
- a static read-only scope
- static checking: cycles detection, undefined variables detection

# Install

```shell
pip install lazycon
```

# Features

## Basic

Let's define a config file `example.config` and see what it can do

```python
# define constants
num_steps = 100
database_table = 'user_data'

# or more complex structures
parameters = {
    'C': 10,
    'metric': 'roc_auc',
}
values_to_try = [0, 1, 2, 3, 4]

# you can use and call builtins
some_range = list(range(100))
# or even comprehensions!
squares = [i ** 2 for i in range(20)]
```

Now let's load our config from python

```python
from lazycon import load

config = load('example.config')
print(config.database_table)
# 'user_data'
```

Need to change an existing config? No problem!

```python
from lazycon import load

config = load('example.config')
config.update(
    database_table='customer_data',
    some_range=[1, 3, 5],
)
config.dump('updated.config')
```

## Advanced

Python-based configs can do so much more! Let's create another `advanced.config`:

```python
# combine config entries
x = 1
y = 2
z = x + y

# define lambdas
callback = lambda value: 1 if value == 0 else (1 / value)


# or more complex functions
def strange_normalize(a, b):
    temp = a ** 2 + b ** 2
    return a / temp, b / temp
```

You can import from other python libraries:

```python
import numpy as np
from math import sqrt

const = np.pi / np.e
proportions = sqrt(2)
```

Or even other configs!

```python
# import from `example.config` defined above
from .example import *

extended_values_to_try = values_to_try + [101, 102]
```

# Contribute

Just get the project from GitHub and modify it how you please!

```shell
git clone https://github.com/maxme1/lazycon.git
cd lazycon
pip install -e .
```
