class Value:
    """
    Stores a single scalar value and its gradient.

    :param data: The value of the scalar
    :param _children: The nodes that this node depends on
    :param _op: The operation that was performed to create this node

    :ivar data: The value of the scalar
    :ivar grad: The gradient of the scalar
    :ivar _prev: The nodes that this node depends on
    :ivar _op: The operation that was performed to create this node
    :ivar _backward: The function that calculates the gradient of this node and the gradients of its children using the
    chain rule
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Calculates the gradient of this node and the gradients of its children using the chain rule.
        """

        topo = self.topological_sort()

        self.grad = 1.0

        for node in topo:
            node._backward()

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
