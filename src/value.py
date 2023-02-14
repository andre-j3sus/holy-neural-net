import math


class Value:
    """
    Stores a single scalar value and its gradient.

    :ivar data: The value of the scalar
    :ivar grad: The gradient of the scalar
    :ivar _prev: The nodes that this node depends on
    :ivar _op: The operation that was performed to create this node
    :ivar _backward: The function that calculates the gradient of this node and the gradients of its children using the
    chain rule
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    def backward(self):
        """
        Calculates the gradient of this node and the gradients of its children using the chain rule.
        First performs a topological sort on the nodes in the computational graph.
        Then changes the gradient of the root node to 1 and propagates the gradients backwards, calling the _backward
        function of each node.
        """

        topo = self.topological_sort()
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    # Operator overloading

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), 'Power must be a number'
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Calculates the hyperbolic tangent of this node.
        """

        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        Calculates the rectified linear unit of this node.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return self - other

    def __rpow__(self, other):  # other ** self
        return self ** other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def topological_sort(self):
        """
        Performs a topological sort on the nodes in the computational graph.
        More about topological sort: https://en.wikipedia.org/wiki/Topological_sorting

        :return: A list of nodes in topological order.
        """
        topo = []
        visited = set()

        def visit(node):
            """Visit a node and its children."""
            if node in visited:
                return

            visited.add(node)
            for child in node._prev:
                visit(child)
            topo.append(node)

        visit(self)
        return topo
