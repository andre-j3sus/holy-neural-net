import math

from src.utils.graph_utils import draw_dot
from src.value import Value


def test_add():
    x = Value(1)
    y = Value(2)
    z = x + y
    assert z.data == 3

    z.backward()
    assert x.grad == 1
    assert y.grad == 1

    draw_dot(z)


test_add()


def test_mul():
    x = Value(2)
    y = Value(3)
    z = x * y
    assert z.data == 6

    z.backward()
    assert x.grad == 3
    assert y.grad == 2


test_mul()


def test_pow():
    x = Value(2)
    y = x ** 3
    assert y.data == 8

    y.backward()
    assert x.grad == 12


test_pow()


def test_tanh():
    x = Value(1)
    y = x.tanh()
    assert y.data == math.tanh(1)

    y.backward()
    assert x.grad == 1 - math.tanh(1) ** 2


test_tanh()


def test_neg():
    x = Value(1)
    y = -x
    assert y.data == -1

    y.backward()
    assert x.grad == -1


test_neg()


def test_radd():
    x = Value(1)
    y = 2 + x
    assert y.data == 3

    y.backward()
    assert x.grad == 1


test_radd()


def test_rmul():
    x = Value(2)
    y = 3 * x
    assert y.data == 6

    y.backward()
    assert x.grad == 3


test_rmul()
