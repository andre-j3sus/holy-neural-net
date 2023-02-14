# holy-neural-net 🧠

> A neural network/machine learning library built with Python. The main goal of this project is to deepen my
> understanding of the field through hands-on implementation and building of different models/algorithms.

> **Note:** This project was inspired by [micrograd](https://github.com/karpathy/micrograd) project, built by Andrej
> Karpathy. The video series on YouTube is also a great resource for learning about neural networks and machine
> learning. I highly recommend
> it: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

---

## Components

### Value

The `Value` class is the core of the library. It is a wrapper around a `float` value that allows for automatic
calculation of gradients. The `Value` class is used to represent the weights and biases of the neural network.

This class is implemented in the `value.py` file, and has operatior overloads for the basic arithmetic operations
(`+`, `-`, `*`, `/`). The `Value` class also has a `backward` method that calculates the gradient of the value and
the gradient of the values that it depends on.

Example:

```python
from src.value import Value

a = Value(3.0)
b = Value(-2.0)
c = a + b
d = a * b + b ** 2
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).tanh()
e = c - d
f = e ** 2
g = f / 2.0
g += 8.0 / f
print(f'{g.data:.4f}')  # prints 312.5105, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints 574.9789, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 299.9821, i.e. the numerical value of dg/db
```
