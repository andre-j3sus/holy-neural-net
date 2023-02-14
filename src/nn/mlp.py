from src.nn.layer import Layer


class MLP:
    """
    Multi-layer perceptron (MLP) is a feedforward artificial neural network model that maps sets of input data onto a
    set of appropriate outputs.
    """

    def __init__(self, num_inputs: int, layers_sizes: list):
        layers = [num_inputs] + layers_sizes
        self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(layers_sizes))]

    def __call__(self, x):
        """
        Calculates the output of the MLP.

        :param x: The input to the MLP.
        :return: The output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x
