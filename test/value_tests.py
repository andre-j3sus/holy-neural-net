from src.value import Value


def test_add():
    x = Value(1)
    y = Value(2)
    z = x + y
    assert z.data == 3

    z.backward()
    assert x.grad == 1
    assert y.grad == 1


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
