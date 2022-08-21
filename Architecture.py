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
        if input_layer is not None:
            assert(input_layer.size() > 0)
            # init_input_weights_1D must be 1D for a simple node
            assert(init_input_weights_1D.shape == (input_layer.size(), 1) or \
                    init_input_weights_1D.shape == (input_layer.size(), ))
        else:
            assert init_input_weights_1D is None,\
                "You must not specify init_input_weights_1D for the input layer."
            assert init_bias is None,\
                "You must not specify init_bias for the input layer."
            
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
        assert self.input_layer is not None,\
            '_coalesced_input cannot be called on the input layer, yet self.input_layer was None.'
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
    to the nodes of the subsequent (downstream) layer.

    The number of inputs to any node is always the size of the previous layer 
    since every one of its nodes' output is one of this node's inputs.

    Layers breakup the provided init_input_weights_2D and construct each node with
    the appropriate column of the matrix.

    A Layer must check if its previous layer is a InjectionLayer because this 
    situation dictates that the init_input_weights_2D be the identity matrix.
    """
    def __init__(self, size: int, init_input_weights_2D: np.array, 
                 init_biases_1D: np.array, input_layer):
        assert(size > 0),\
            f"layer sizes must be strictly positive, not ({size})"

        if input_layer is not None:
            assert(len(init_input_weights_2D.shape) == 2)
            assert init_input_weights_2D.shape == (input_layer.size(), size),\
                'init_input_weights_2D must be a matrix of dimension ({}, {}) not {}!'\
                .format(input_layer.size(), size, init_input_weights_2D.shape)
            assert init_biases_1D.shape == (size,),\
                'init_biases_1D extents {} must match layer size {}'\
                .format(init_biases_1D.shape[0], size)
        else:
            assert init_input_weights_2D is None,\
                "You must not specify init_input_weights_2D for the input layer."
            assert init_biases_1D is None,\
                "You must not specify init_biases_1D for the input layer."




        self.input_layer = input_layer

        self.nodes = []
        if input_layer is not None:
            for n in range(size):
                self.nodes.append(Node(init_input_weights_2D[:,n], init_biases_1D[n], input_layer))
        else:
            for n in range(size):
                self.nodes.append(Node(None, None, None))

        self.global_input_values = None

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
        print(delta_weights.shape)
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
        if self.input_layer is not None:
            return np.array([n._coalesced_input() for n in self.nodes])
        else:
            return self.global_input_values

    def outputs(self) -> list[np.float32]:
        if self.input_layer is None:
            # Forget the nodes and just compute the layer output array directly
            return np.vectorize(sigmoid)(self.global_input_values)
        else:
            # Doesn't this seem clunky and possibly slow?
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

class Network():
    """
    Holds the list of Layers that defines the network. Also provides convenient
    initialization and update methods.
    """
    def __init__(self, layer_sizes: tuple):
        assert all([size > 0 for size in layer_sizes])

        self.layers = []
        # Insert Layer 0
        self.layers.append(Layer(layer_sizes[0], 
                                 None,
                                 None,
                                 None))
        # Insert all the rest
        # Skip the first, already constructed, entry
        for idx, size in enumerate(layer_sizes):
            if(idx >= 1):
                self.layers.append(Layer(size, 
                                         random_array((layer_sizes[idx - 1], size)),
                                         random_array((size,)),
                                         self.layers[idx - 1]))

    def layer_sizes(self) -> tuple:
        layer_sizes = [layer.size() for layer in self.layers]
        return tuple(layer_sizes)

    def num_layers(self) -> int:
        return len(self.layer_sizes())

    def outputs(self) -> np.array:
        return self.layers[-1].outputs()

    def print_status(self, i: int, example: dict):
        result = self.outputs()
        cost = network.cost(example['labels'], result)
        print(i, result, cost)

    def adjust_global_input_values(self, global_input_values: np.array):
        assert len(global_input_values.shape) == 1,\
            f"global_input_values' rank must be 1 not ({len(global_input_values.shape)})."
        layer_sizes = self.layer_sizes()
        assert layer_sizes[0] == global_input_values.size,\
            f'Network(): global_input_values size ({global_input_values.size})'\
            +'does not match layer 0 spec size ({layer_sizes[0]}).'
        # Only apply the global_input_values to the 0th layer
        self.layers[0].global_input_values = global_input_values

    def add_delta_biases(self, delta_biases_2D: list):
        # The zeroth layer has no biases. Thus the first element of 
        # delta_biases_2D is ignored.
        ranks = [0 if delta_biases[l] is None else len(delta_biases[l].shape) for l in range(self.num_layers())]
        rankish = map(lambda x: x <= 1, ranks)
        assert all(rankish),\
            'Each array in delta_biases_2D must be None or rank 1, not {}.'\
            .format((len(delta_biases_2D[l].shape)))

        dims = [-1 if delta_biases[l] is None else delta_biases[l].size for l in range(self.num_layers())]
        dimish = [x == y for x,y in zip(dims, self.layer_sizes())]
        assert all(dimish[1:]),\
            'delta_biases_2D sizes {} must match layer sizes {}.'\
            .format(([dims[l] for l in delta_biases_2D[1:]]), self.layer_sizes()[1:])
        for l, layer in enumerate(self.layers):
            if(l>=1):
                layer.add_delta_biases(delta_biases_2D[l])

    def add_delta_weights(self, delta_weights: list):
        # The zeroth layer has no weights or biases. Thus the first element of 
        # delta_weights is ignored.
        assert len(delta_weights) == len(self.layers),\
            f'Length of delta_weights {len(delta_weights)} must match number of layers {len(self.layers)}.'
        for idx, layer_matrix in enumerate(delta_weights):
            if(idx >= 1):
                self.layers[idx].add_delta_weights(layer_matrix)

    def random_delta_biases(self) -> list:
        # The zeroth layer has no biases. Thus the first element of 
        # delta_biases is left empty.
        result = [np.empty(())]
        for l in range(len(self.layer_sizes())):
            if(l >= 1):
                result.append(random_array((self.layer_sizes()[l],)))
        return result

    def random_delta_weights(self) -> list:
        # The zeroth layer has no weights and no biase. Thus the first element of delta_weights is 
        # left empty.
        result = [None]
        for idx, size in enumerate(self.layer_sizes()):
            if(idx >= 2):
                result.append(random_array((self.layer_sizes()[idx - 1], size)))
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

    def deriv_z_wrt_a_m1(self, layer_id: int) -> np.array:
        """
        Computes the dervative of 'z^l_n' in terms of the previous layer's
        post-activation node outputs 'a^(l-1)_m' which turns out to be the 
        weights 'W^l_{m n}'

        layer_id: int - 'l', which indicates to which layer these quantities 
            belong
        """        
        num_nodes_l = self.layer_sizes()[layer_id]
        num_nodes_lm1 = self.layer_sizes()[layer_id - 1]
        result = np.empty((num_nodes_lm1, num_nodes_l), dtype=np.float32)
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

        labels: np.array - ground truth values for output
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

        labels: np.array - ground truth values for output
        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        """
        num_nodes_f = self.layer_sizes()[-1]
        return -learning_rate * np.einsum('n, n, n -> n',
                                          self.deriv_Cost_wrt_a_output(labels),
                                          self.deriv_a_wrt_z(-1),
                                          self.deriv_z_wrt_b(-1))

    def _compute_back_prop_monomer_ending_at_l(self, l: int) -> np.array:
        """
        Computes the pair of partial derivs that form a repeated monomer when
        calculating delta_weights and delta_biases for interior layers.

        l: int - The monomer crosses layers. This parameter designates the 
            upstream target layer, not the source layer.
        """
        return np.einsum('nm,n -> nm',
                         self.deriv_z_wrt_a_m1(l+1), 
                         self.deriv_a_wrt_z(l), dtype=np.float32)

    def _compute_delta_weights_at_layer_l(self, target_layer_id: int, labels: np.array) -> np.array:
        """
        Computes delta_weights for any layer, excluding the 0th, of course.

        target_layer_id: int - the layer number of the layer owning the target weights
        labels: np.array - ground truth values for output
        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        """
        layer_sizes = self.layer_sizes()
        output_layer_id = len(layer_sizes) - 1

        assert target_layer_id != 0,\
            'Cannot call _compute_delta_weights_at_layer_l on the input layer.\n'\
            +f'Input layer has no weights to update. target_layer_id was {target_layer_id}.'
        assert target_layer_id > 0 and target_layer_id <= output_layer_id,\
            f'target_layer_id must be > 0 and <= {output_layer_id}, not {target_layer_id}.'

        #   This needs to be improved. We don't want to repeatedly calling outputs().
        # Initiailize the chain of matrices to be multiplied
        chain_of_partials = [np.einsum('m,m -> m',
                                      self.deriv_Cost_wrt_a_output(labels, self.outputs()),
                                      self.deriv_a_wrt_z(output_layer_id))]
        for l in range(output_layer_id-1, target_layer_id-1, -1):
            chain_of_partials.append(self._compute_back_prop_monomer_ending_at_l(l).T)

        if len(chain_of_partials) > 1:
            return np.outer(np.linalg.multi_dot(chain_of_partials), self.deriv_z_wrt_weights(target_layer_id)).T
        else:
            return np.outer(chain_of_partials[0], self.deriv_z_wrt_weights(target_layer_id)).T


    def compute_delta_weights(self, labels: np.array, learning_rate: np.float32) -> list:
        """
        Computes delta_weights for all layers, excluding the 0th, of course.

        labels: np.array - ground truth values for output
        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        
        returns a list of matrices
        """
        output_layer_id = len(self.layer_sizes()) - 1
        result = [None]
        for l in range(1, output_layer_id+1):
            result.append(-learning_rate * self._compute_delta_weights_at_layer_l(l, labels))
        return result





# ============================================
def all_zeros_array(shape: tuple) -> np.array:
    return np.zeros(shape, dtype=np.float32)

def all_ones_array(shape: tuple) -> np.array:
    return np.ones(shape, dtype=np.float32)

def tan_random_float() -> np.float32: 
    return (lambda x: tan(2.*pi*(x - 0.5)))(R.random())

def dustin_random_float() -> np.float32:
    return 12.*R.random() - 6.

def random_array(shape: tuple) -> np.array:
    ranlist = []
    for _ in range(prod(shape)):
        ranlist.append(tan_random_float())
#        ranlist.append(dustin_random_float)
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
    emx:np.float64 = exp(-input)
    assert emx > 0.,\
        f'deriv_sig: exp(-x) should be strictly positive, not {emx}'
    return np.float32(- emx/((1. + emx)*(1. + emx)))


def char(C: int) -> str:
    return chr(C + ord('a'))




### ==================================================
### Construct example
R = Random()
R.seed(1234)
input_values = np.array([0., -72.0, 0.26, 8.0])
label_values = np.array([0.7, 0.2])
example = {'features': input_values,
           'labels': label_values}

### Construct network
layer_sizes = (4,3,2)
network = Network(layer_sizes)

learning_rate = 0.1

### Set inputs
network.adjust_global_input_values(example['features'])
network.print_status(0, example)

for iteration in range(1, 10):
    print(f'iteration {iteration} --- final layer delta weights calcs:')
    delta_weights = network.compute_delta_weights(example['labels'], learning_rate)

    delta_biases_1 = all_zeros_array((6,))
    delta_biases_2 = network.compute_delta_biases_f_layer(example['labels'], learning_rate)
    delta_biases = [None, delta_biases_1, delta_biases_2]
#    print('delta_weights_f_layer', delta_weights, delta_biases)

    network.add_delta_weights(delta_weights)
    # network.add_delta_biases(delta_biases)
    network.print_status(iteration, example)


