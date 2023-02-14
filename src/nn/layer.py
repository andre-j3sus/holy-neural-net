from src.nn.neuron import Neuron


class Layer:
    """
    A layer is a collection of neurons that are connected to each other.

    :ivar neurons: The neurons in the layer.
    """

    def __init__(self, num_neuron_inputs: int, size: int):
        self.neurons = [Neuron(num_neuron_inputs) for _ in range(size)]

    def __call__(self, x):
        """
        Calculates the output of the layer.

        :param x: The input to the layer.
        :return: The output of the layer.
        """
        return [neuron(x) for neuron in self.neurons]
