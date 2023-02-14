import random

from src.value import Value


class Neuron:
    """
    Represents a neuron in a neural network.

    :ivar weights: The weights of each neuron input.
    :ivar bias: The bias of the neuron. This is the value that is added to the sum of the inputs.
    """

    def __init__(self, num_inputs: int):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Calculates the output of the neuron.

        :param x: The input to the neuron.
        :return: The output of the neuron.
        """
        return sum((xi * wi for xi, wi in zip(self.weights, x)), self.bias).relu()
