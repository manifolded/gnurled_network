import numpy as np
from math import exp, tan, pi, prod
from numpy.random import rand

class Node():
    """
    Nodes are the vertices in the network. They provide a minimal unit of 
    computation. Nodes are combined to make layers, see class Layers for 
    more detail. 

    The outputs of the nodes of the layer immediately upstream provide
    each node's inputs. These inputs are weighted before being combined into a 
    single output scalar. 

    By design nodes own the weights for their connections with the nodes of the 
    directly upstream layer. They do not own the weights for their downstream 
    connections which are owned instead for the nodes of the next layer.
    """
    def __init__(self, init_input_weights_1D: np.array, init_bias: np.float32, input_layer):
        assert(input_layer.size() > 0)
        # init_input_weights_1D must be 1D for a simple node
        assert(init_input_weights_1D.shape == (input_layer.size(), 1) or \
                init_input_weights_1D.shape == (input_layer.size(), ))
            
        self.input_layer = input_layer
        self.input_weights = init_input_weights_1D
        self.bias = init_bias

    def add_delta_weights(self, delta_weights: np.array):
        assert len(delta_weights.shape) == 1,\
            'delta_weights rank must be 1, not {}'\
            .format(len(delta_weights.shape))
        assert delta_weights.shape[0] == self.input_layer.size(),\
            'Node: Size of previous layer, {}, must match size of delta_weights, {}.'\
            .format(self.input_layer.size(), delta_weights.shape[0])
        self.input_weights += delta_weights

    def output(self) -> np.float32:
        return sigmoid(sum(map(lambda x,y: x*y, self.input_weights, self.input_layer.outputs())) + self.bias)


class Layer():
    """
    Layers are ranks of nodes of any length. These nodes all receive their inputs
    from the nodes of the previous (upstream) layer, and output their computations
    to the nodes of the next (downstream) layer.

    The number of inputs to any node is always the size of the previous layer 
    since every one of its nodes' output is one of this node's inputs.

    Layers breakup the provided init_input_weights_2D and construct each node with
    the appropriate column of the matrix.

    A Layer must check if its previous layer is a InjectionLayer because this 
    situation dictates that the init_input_weights_2D be the identity matrix.
    """
    def __init__(self, size: int, init_input_weights_2D: np.array, init_biases_1D: np.array, input_layer):
        assert(size > 0)
        assert(len(init_input_weights_2D.shape) == 2)
    
        # If the immediately upstream layer is a InjectionLayer, require that it 
        # have same size as this layer.
        if isinstance(input_layer, InjectionLayer):
            assert input_layer.size() == size,\
            'upstream is InjectionLayer -> sizes must match'
        assert init_input_weights_2D.shape == (input_layer.size(), size),\
            'init_input_weights_2D must be a matrix of dimension ({}, {}) not {}!'\
            .format(input_layer.size(), size, init_input_weights_2D.shape)
        assert init_biases_1D.shape == (size,),\
            'init_biases_1D extents {} must match layer size {}'\
            .format(init_biases_1D.shape[0], size)

        self.input_layer = input_layer
        self.nodes = []
        for n in range(size):
            self.nodes.append(Node(init_input_weights_2D[:,n], init_biases_1D[n], input_layer))

    def size(self) -> int:
        return len(self.nodes)

    def add_delta_weights(self, delta_weights: np.array):
        assert len(delta_weights.shape) == 2,\
            'delta_weights rank must be 2, not {}'\
            .format(len(delta_weights.shape))
        assert delta_weights.shape[1] == self.size(),\
            'Number of nodes {} must match delta_weights\' 2nd extents {}'\
            .format(self.size(), delta_weights.shape[1])
        assert delta_weights.shape[0] == self.input_layer.size(),\
            'Number of nodes in previous layer {} must match delta_weight\'s 1st extents {}'\
            .format(delta_weights.shape[0], self.input_layer.size())
        for idx, node in enumerate(self.nodes):
            node.add_delta_weights(delta_weights[:,idx])

    def outputs(self) -> list[np.float32]:
        return np.array([node.output() for node in self.nodes])


class InjectionLayer():
    """ 
    An injection layer is a layer that produces pre-set values. It is intended 
    to serve in the role of a fake extra input_layer that provides values to 
    the actual input layer. While it pretends to be a 'Layer' it has no nodes 
    and only spits out exactly what you last specified with the 
    adjust_global_input_values method.
    """
    def __init__(self, size: int):
        # construct empty global_input_values
        self.global_input_values = np.array([None]*size)
        self._size = size

    def size(self) -> int:
        return self._size

    def outputs(self) -> list[np.float32]:
        return self.global_input_values

    def adjust_global_input_values(self, global_input_values: np.array):
        assert np.linalg.matrix_rank(global_input_values) == 1
        assert global_input_values.shape[0] == self._size
        self.global_input_values = global_input_values

    # To be furnished for the construction of the immediately downstream genuine
    # input layer's input weights, which should be held constant.
    def gen_diag_weights_2D(self) -> np.array:
        return np.identity(self._size)


class Network():
    """
    Holds the list of Layers that defines the network. Also provides convenient
    initialization and update methods.
    """
    def __init__(self, layer_sizes: tuple):
        assert all([size > 0 for size in layer_sizes])
        # Prepend leading InjectionLayer which is hidden from the user
        #   Note that both the 0th and 1st layers are required to have the same size
        true_layer_sizes = (layer_sizes[0],) + layer_sizes

        self.layers = []
        # Insert layer 0
        self.layers.append(InjectionLayer(true_layer_sizes[0]))
        # Insert Layer 1
        self.layers.append(Layer(true_layer_sizes[1], 
                                 self.layers[0].gen_diag_weights_2D(),
                                 all_zeros_array((true_layer_sizes[1],)),
                                 self.layers[0]))
        # Insert all the rest
        # Skip the first two entries that have already been constructed.
        for idx, size in enumerate(true_layer_sizes):
            if(idx >= 2):
                self.layers.append(Layer(size, 
                                        random_array((true_layer_sizes[idx - 1], size)),
                                        random_array((size,)),
                                        self.layers[idx - 1]))

    def layer_sizes(self) -> tuple:
        true_layer_sizes = [layer.size() for layer in self.layers]
        return tuple(true_layer_sizes[1:])

    def outputs(self) -> np.array:
        return self.layers[-1].outputs()

    def adjust_global_input_values(self, global_input_values: np.array):
        assert np.linalg.matrix_rank(global_input_values) == 1
        assert global_input_values.shape[0] == self.layers[0].size()
        self.layers[0].adjust_global_input_values(global_input_values)

    def add_delta_weights(self, delta_weights: list):
        # The zeroth layer has no weights. The first layer has fixed weights, 
        # so no delta. Thus the first two elements of delta_weights are 
        # ignored.
        assert len(delta_weights) == len(self.layers),\
            'Length of delta_weights must match number of layers, they were {} and {} respectively.'\
            .format(len(self.layers), len(delta_weights))
        for idx, layer_matrix in enumerate(delta_weights):
            if(idx >= 2):
                self.layers[idx].add_delta_weights(layer_matrix)

    def random_delta_weights(self) -> list:
        # The zeroth layer has no weights. The first layer has fixed weights, 
        # so no delta. Thus the first two elements of delta_weights are 
        # left empty.
        result = [np.empty(()), np.empty(())]
        layer_sizes = self.layer_sizes()
        true_layer_sizes = (layer_sizes[0],) + layer_sizes
        for idx, size in enumerate(true_layer_sizes):
            if(idx >= 2):
                result.append(random_array((true_layer_sizes[idx - 1], size)))
        return result


def all_zeros_array(shape: tuple) -> np.array:
    return np.zeros(shape, dtype=np.float32)

def all_ones_array(shape: tuple) -> np.array:
    return np.ones(shape, dtype=np.float32)

def tan_random_float() -> np.float32: 
    return (lambda x: tan(2.*pi*(x - 0.5)))(rand())

def random_array(shape: tuple) -> np.array:
    ranlist = []
    for _ in range(prod(shape)):
        ranlist.append(tan_random_float())
    # https://opensourceoptions.com/blog/10-ways-to-initialize-a-numpy-array-how-to-create-numpy-arrays/
    return np.array(ranlist).reshape(shape)

def sigmoid(input: np.float32) -> np.float32:
    """
    Implementation of the classic node activation function called the sigmoid 
    function, $\\frac{1}{1 + e^{-x}}$. Despite the fact that I will inevitably 
    be coding up the derivative before long, I don't think I'm going to make 
    this a class. I don't want the overhead of a constructor.
    """
    return 1./(1. + exp(- input))


layer_sizes = (3,6,2)
input_values = np.array([1.0, -0.5, 2.0])

network = Network(layer_sizes)
network.adjust_global_input_values(input_values)
print([val for val in network.outputs()])

delta_weights = network.random_delta_weights()
network.add_delta_weights(delta_weights)
print([val for val in network.outputs()])
