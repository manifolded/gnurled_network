from platform import node
from random import Random
# from xml.dom.minicompat import NodeList
import numpy as np
from math import exp, tan, pi, prod
# from numpy.random import rand

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

    def add_delta_bias(self, delta_bias: np.float32):
        self.bias += delta_bias

    def add_delta_weights(self, delta_weights: np.array):
        assert len(delta_weights.shape) == 1,\
            'delta_weights rank must be 1, not {}'\
            .format(len(delta_weights.shape))
        assert delta_weights.shape[0] == self.input_layer.size(),\
            'Node: Size of previous layer, {}, must match size of delta_weights, {}.'\
            .format(self.input_layer.size(), delta_weights.shape[0])
        self.input_weights += delta_weights

    def _coalesced_input(self) -> np.float32:
        """
        Returns the 'pre-activation output' aka 'coalesced input' of this 
        node.
        """
        return sum(map(lambda x,y: x*y, self.input_weights, self.input_layer.outputs())) + self.bias

    def output(self) -> np.float32:
        return sigmoid(self._coalesced_input())

    def cost(self, label: np.float32, output: np.float32 = None) -> np.float32:
        # Avoid unnecessarily recomputing output by passing as arg
        out = output if output is not None else self.output()
        delta = label - out
        return delta * delta

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

    def add_delta_biases(self, delta_biases_1D: np.array):
        assert len(delta_biases_1D.shape) == 1,\
            'Rank of delta_biases_1D must be 1, not {}'\
            .format(len(delta_biases_1D.shape))
        assert delta_biases_1D.shape[0] == self.size(),\
            'Dim of delta_biases_1D {} must equal number of nodes {}.'\
            .format(delta_biases_1D.shape[0], self.size())
        for n, node in enumerate(self.nodes):
            node.add_delta_bias(delta_biases_1D[n])

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

    def _coalesced_inputs(self) -> np.array:
        """
        Returns an array of the 'pre-activation outputs' aka 'coalesced 
        inputs' of the nodes in this layer.
        """
        return np.array([n._coalesced_input() for n in self.nodes])

    def outputs(self) -> list[np.float32]:
        return np.array([node.output() for node in self.nodes])

    def cost(self, labels: np.array, outputs: np.array = None) -> np.float32:
        assert len(labels.shape) == 1,\
            'Labels rank must be 1, not {}'.format(len(labels.shape))
        assert labels.shape[0] == self.size(),\
            'Labels must have the same number of elements {} as there are nodes in this layer {}.'\
            .format(labels.shape[0], self.size())
        # Avoid unnecessarily recomputing output by passing as arg
        outs = outputs if outputs is not None else self.outputs()
        return sum([node.cost(labels[n], outs[n]) for n, node in enumerate(self.nodes)])

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

    def num_layers(self) -> int:
        return len(self.layer_sizes())

    def outputs(self) -> np.array:
        return self.layers[-1].outputs()

    def print_status(self, i: int, example: dict):
        result = self.outputs()
        cost = network.cost(example['labels'], result)
        print(i, result, cost)

    def adjust_global_input_values(self, global_input_values: np.array):
        assert np.linalg.matrix_rank(global_input_values) == 1
        assert global_input_values.shape[0] == self.layers[0].size()
        self.layers[0].adjust_global_input_values(global_input_values)

    def add_delta_biases(self, delta_biases_2D: list):
        # The zeroth layer has no biases. Thus the first element of 
        # delta_biases_2D is ignored.
        ranks = [0 if delta_biases[l] is None else len(delta_biases[l].shape) for l in range(1, self.num_layers())]
        rankish = map(lambda x: x <= 1, ranks)
        assert all(rankish),\
            'Each array in delta_biases_2D must be None or rank 1, not {}.'\
            .format((len(delta_biases_2D[l].shape)))

        true_layer_sizes = (self.layer_sizes()[0],) + self.layer_sizes()
        dims = [-1 if delta_biases[l] is None else delta_biases[l].size for l in range(self.num_layers())]
        dimish = [x == y for x,y in zip(dims, true_layer_sizes)]
        print('add_delta_biases: ', dims)
        assert all(dimish[1:]),\
            'delta_biases_2D sizes {} must match layer sizes {}.'\
            .format(([dims[l] for l in delta_biases_2D[1:]]), self.layer_sizes())
        for l, layer in enumerate(self.layers):
            if(l>=1):
                layer.add_delta_biases(delta_biases_2D[l])

    def add_delta_weights(self, delta_weights: list):
        # The zeroth layer has no weights. The first layer has fixed weights, 
        # so no delta. Thus the first two elements of delta_weights are 
        # ignored.
        assert len(delta_weights) == len(self.layers),\
            'Length of delta_weights {} must match number of layers {}.'\
            .format(len(self.layers), len(delta_weights))
        for idx, layer_matrix in enumerate(delta_weights):
            if(idx >= 2):
                self.layers[idx].add_delta_weights(layer_matrix)

    def random_delta_biases(self) -> list:
        # The zeroth layer has no biases. Thus the first element of 
        # delta_biases is left empty.
        result = [np.empty(())]
        layer_sizes = self.layer_sizes()
        true_layer_sizes = (layer_sizes[0],) + layer_sizes
        for idx, size in enumerate(true_layer_sizes):
            if(idx >= 1):
                result.append(random_array((true_layer_sizes[idx],)))
        return result

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

    # ============================
    # Toolkit for Back-Propagation 
    def cost(self, labels: np.array, outputs: np.array = None) -> np.float32:
        assert len(labels.shape) == 1,\
            'Labels rank must be 1, not {}'.format(len(labels.shape))
        final_layer_size = self.layer_sizes()[-1]
        assert labels.shape[0] == final_layer_size,\
            'Labels must have the same number of elements {} as there are nodes in the final layer {}.'\
            .format(labels.shape[0], final_layer_size)

        # Avoid recomputing the outputs by passing in the value with the 2nd argument
        outs = outputs if outputs is not None else self.outputs()
        return self.layers[-1].cost(labels, outs)
        
    def deriv_Cost_wrt_a_output(self, labels: np.array, outputs: np.array = None) -> np.array:
        """
        Computes the derivative of the cost function with respect to the 
        network's outputs.

        labels: np.array - ground truth results expected
        outputs: np.array - the network's prediction
        """
        outs = outputs if outputs is not None else self.outputs()
        return np.array([-((label/out) + (1. - label)/(1. - out)) for label, out in zip(labels, outs)])

    def deriv_a_wrt_z(self, layer_id: int) -> np.array:
        """
        Computes the derivative of the network's outputs with respect to the 
        nodes' coalesced inputs, often denoted as 'z^l_n'.

        layer_id: int - indicates which layer's outputs to be differentiated
        z_layer_inputs: np.array - contains the PRE-activation node values for 
        layer 'l'. These may be known either as coalesced inputs or 
        pre-activation outputs. It means the same thing. Designated z^l_n to 
        distinguish them from layer activations, always designated a^l_n.
        """
        return np.array([deriv_sig(z) for z in self.layers[layer_id]._coalesced_inputs()])

    def deriv_z_wrt_weights(self, layer_id: int) -> np.array:
        """
        Computes the derivative of 'z^l_n' in terms of the weights, 'W^l_{m n}'
        which turns out to be nothing more than the post-activation ouputs of 
        layer 'l'. 

        layer_id: int - is 'l' which indicates to which layer these quantities belong
        """
        # Note that output is strangely independent of the layer
        #    node index. Turns out it must be. 

        #### Should I add in the usual input caching check? Not for an internal layer? ####
        return self.layers[layer_id - 1].outputs()

    def deriv_z_wrt_a(self, layer_id: int) -> np.array:
        """
        Computes the dervative of 'z^l_n' in terms of the previous layer's
        post-activation node outputs 'a^(l-1)_m' which turns out to be the 
        weights 'W^l_{m n}'

        layer_id: int - 'l', which indicates to which layer these quantities 
            belong
        """        
        num_nodes_l = self.layer_sizes()[layer_id]
        num_nodes_lm1 = self.layer_sizes()[layer_id - 1]
        result = np.array((num_nodes_lm1, num_nodes_l), dtype=np.float32)
        for n in range(num_nodes_l):
            result[:,n] = self.layers[layer_id].nodes[n].input_weights
        return result

    def deriv_z_wrt_b(self, layer_id: int) -> np.array:
        """
        Computes the derivative of 'z^l_n' with respect to the layer's biases
        'b^l_n', which turns out to be just ones.
        """
        return all_ones_array((self.layer_sizes()[layer_id],))

    def compute_delta_weights_f_layer(self, labels: np.array, learning_rate: np.float32) -> np.array:
        """
        Computes delta_weights for the output layer.

        learning_rate: np.float32 - arbitrary coefficient for delta_weights
        """
        num_nodes_f = self.layer_sizes()[-1]
        num_nodes_fm1 = self.layer_sizes()[-2]
        result = np.array((num_nodes_fm1, num_nodes_f), dtype=np.float32)

        return -learning_rate * np.einsum('n, n, m -> m n', 
                                          self.deriv_Cost_wrt_a_output(labels),
                                          self.deriv_a_wrt_z(-1),
                                          self.deriv_z_wrt_weights(-1))

    def compute_delta_biases_f_layer(self, labels: np.array, learning_rate: np.float32) -> np.array:
        """
        Computes delta_biases for the output layer.

        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        """
        num_nodes_f = self.layer_sizes()[-1]
        return -learning_rate * np.einsum('n, n, n -> n',
                                          self.deriv_Cost_wrt_a_output(labels),
                                          self.deriv_a_wrt_z(-1),
                                          self.deriv_z_wrt_b(-1))


# ============================================
def all_zeros_array(shape: tuple) -> np.array:
    return np.zeros(shape, dtype=np.float32)

def all_ones_array(shape: tuple) -> np.array:
    return np.ones(shape, dtype=np.float32)

def tan_random_float() -> np.float32: 
    return (lambda x: tan(2.*pi*(x - 0.5)))(R.random())

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
    # Overflows will often show up here. There's no reason to try and catch
    # them. It's natural for the code to crash when it's wildly diverging.
    return 1./(1. + exp(- input))

def deriv_sig(input: np.float32) -> np.float32:
    emx:np.float64 = - exp(input)
    return np.float32(- emx/((1. + emx)*(1. + emx)))


### ==================================================
### Construct example
R = Random()
R.seed(8397459)
input_values = np.array([1.0, -0.5, 2.0])
label_values = np.array([0.7, 1.1])
example = {'features': input_values,
           'labels': label_values}

### Construct network
layer_sizes = (3,6,2)
network = Network(layer_sizes)

learning_rate = 0.1

### Set inputs
network.adjust_global_input_values(example['features'])
network.print_status(0, example)

for iteration in range(1, 10):
    print(f'iteration {iteration} --- final layer delta weights calcs:')
    # print('deriv_Cost_wrt_a_output:', network.deriv_Cost_wrt_a_output(label_values))
    # print('deriv_a_wrt_z:', network.deriv_a_wrt_z(-1))
    # print('deriv_z_wrt_weights:', network.deriv_z_wrt_weights(-1))
    delta_weights_1 = all_zeros_array((3,6))
    delta_weights_2 = network.compute_delta_weights_f_layer(example['labels'], learning_rate)
    delta_weights = [None, None, delta_weights_1, delta_weights_2]
    delta_biases_0 = all_zeros_array((3,))
    delta_biases_1 = all_zeros_array((6,))
    delta_biases_2 = network.compute_delta_biases_f_layer(example['labels'], learning_rate)
    delta_biases = [None, delta_biases_0, delta_biases_1, delta_biases_2]
#    print('delta_weights_f_layer', delta_weights, delta_biases)

    network.add_delta_weights(delta_weights)
    network.add_delta_biases(delta_biases)
    network.print_status(iteration, example)


