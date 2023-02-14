from graphviz import Digraph
from src.value import Value

"""
Contains utility functions for drawing graphs using graphviz.
Use the draw_dot function to draw a graph from a root node.

Example:
a = Value(1)
b = Value(2)
c = a + b
draw_dot(c).render('test.gv', view=True)
"""


def trace(root: Value):
    """
    Creates a set of nodes and edges from a root node.

    :param root: The root node of the graph.
    :return: A tuple of (nodes, edges)
    """

    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value, format='svg', rankdir='LR'):
    """
    Draws a graph from a root node.

    :param root: The root node of the graph.
    :param format: The format of the output graph (png | svg | ...).
    :param rankdir: The direction of the graph (TB (top to bottom graph) | LR (left to right)).

    :return: A graphviz Digraph object
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
