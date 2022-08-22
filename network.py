from random import Random
import numpy as np
from math import exp, tan, pi, prod, log
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
            assert init_input_weights_2D.shape == (input_layer.size, size),\
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

        self.size = size
        self.input_layer = input_layer
        self.input_weights = init_input_weights_2D
        self.biases = init_biases_1D
        self.global_input_values = None

    def size(self) -> int:
        return self.size

    def add_delta_weights_and_biases(self, delta_weights_and_biases: tuple):
        delta_weights = delta_weights_and_biases[0]
        assert len(delta_weights.shape) == 2,\
            'delta_weights rank must be 2, not {}'\
            .format(len(delta_weights.shape))
        assert delta_weights.shape[1] == self.size,\
            'Number of nodes {} must match delta_weights\' 2nd extents {}'\
            .format(self.size(), delta_weights.shape[1])
        assert delta_weights.shape[0] == self.input_layer.size,\
            'Number of nodes in previous layer {} must match delta_weight\'s '\
            +'second index extents {}'\
            .format(self.input_layer.size(), delta_weights.shape[0])

        delta_biases = delta_weights_and_biases[1]
        assert len(delta_biases.shape) == 1,\
            'Rank of delta_biases must be 1, not {}'\
            .format(len(delta_biases.shape))
        assert delta_biases.shape[0] == self.size,\
            'Dim of delta_biases {} must equal number of nodes {}.'\
            .format(delta_biases.shape[0], self.size)

        self.input_weights += delta_weights
        self.biases += delta_biases
        
    def _coalesced_inputs(self) -> np.array:
        """
        Returns an array of the 'pre-activation outputs' aka 'coalesced 
        inputs' of the nodes in this layer.
        """
        return self.global_input_values if self.input_layer is None else \
            np.dot(self.input_weights.T, self.input_layer.outputs()) + self.biases


    def outputs(self) -> np.array:
        return np.vectorize(Activation.sigmoid)(self._coalesced_inputs())

class Network():
    """
    Holds the list of Layers that defines the network. Also provides convenient
    initialization and update methods.
    """
    def __init__(self, layer_sizes: tuple, array_generator: callable, cost_implementation: callable):
        assert all([size > 0 for size in layer_sizes])

        self.cost_implementation = cost_implementation

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
                                         array_generator((layer_sizes[idx - 1], size)),
                                         array_generator((size,)),
                                         self.layers[idx - 1]))

    def layer_sizes(self) -> tuple:
        layer_sizes = [layer.size for layer in self.layers]
        return tuple(layer_sizes)

    def num_layers(self) -> int:
        return len(self.layer_sizes())

    def outputs(self) -> np.array:
        return self.layers[-1].outputs()

    def print_status(self, iteration: int, example: list):
        result = self.outputs()
        cost = self.cost(example[1], result)
        print(iteration, result, cost)

    def adjust_global_input_values(self, global_input_values: np.array):
        assert len(global_input_values.shape) == 1,\
            f"global_input_values' rank must be 1 not ({len(global_input_values.shape)})."
        layer_sizes = self.layer_sizes()
        assert layer_sizes[0] == global_input_values.size,\
            f'Network(): global_input_values size ({global_input_values.size})'\
            +'does not match layer 0 spec size ({layer_sizes[0]}).'
        # Only apply the global_input_values to the 0th layer
        self.layers[0].global_input_values = global_input_values

    def add_delta_weights_and_biases(self, delta_weights_and_biases: list):
        # The zeroth layer has no weights or biases. Thus the 0th element of 
        # delta_weights_and_biases is ignored.
        assert len(delta_weights_and_biases) == self.num_layers(),\
            f'len(delta_weights_and_biases ({len(delta_weights_and_biases)})'\
                f'must equal num_layers ({self.num_layers()})' 
        for idx in range(len(delta_weights_and_biases)):
            if(idx >= 1):
                self.layers[idx].add_delta_weights_and_biases(delta_weights_and_biases[idx])

    def random_delta_weights_and_biases(self) -> list:
        # The zeroth layer has no weights or biases, thus leave the 0th element
        # empty.
        result = [(None, None)]
        layer_sizes = self.layer_sizes()
        for idx, size in enumerate(layer_sizes):
            if(idx >= 1):
                result.append((RandomUtils.random_array((layer_sizes[idx - 1], size)),  
                               RandomUtils.random_array((size,))))
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
        return self.cost_implementation.cost(labels, outs)
        
    def _deriv_Cost_wrt_a_output(self, labels: np.array, outputs: np.array = None) -> np.array:
        """
        Computes the derivative of the cost function with respect to the 
        network's outputs.

        labels: np.array - ground truth results expected
        outputs: np.array - the network's prediction
        """
        outs = outputs if outputs is not None else self.outputs()
        return self.cost_implementation.cost_deriv(labels, outs)

    def _deriv_a_wrt_z(self, layer_id: int) -> np.array:
        """
        Computes the derivative of the network's outputs with respect to the 
        nodes' coalesced inputs, often denoted as 'z^l_n'.

        layer_id: int - indicates which layer's outputs to be differentiated
        z_layer_inputs: np.array - contains the PRE-activation node values for 
        layer 'l'. These may be known either as coalesced inputs or 
        pre-activation outputs. It means the same thing. Designated z^l_n to 
        distinguish them from layer activations, always designated a^l_n.
        """
        return np.array([Activation.deriv_sig(z) for z in self.layers[layer_id]._coalesced_inputs()])

    def _deriv_z_wrt_weights(self, layer_id: int) -> np.array:
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

    def _deriv_z_wrt_a_m1(self, layer_id: int) -> np.array:
        """
        Computes the dervative of 'z^l_n' in terms of the previous layer's
        post-activation node outputs 'a^(l-1)_m' which turns out to be the 
        weights 'W^l_{m n}'

        layer_id: int - 'l', which indicates to which layer these quantities 
            belong
        """        
        return self.layers[layer_id].input_weights

    def _deriv_z_wrt_b(self, layer_id: int) -> np.array:
        """
        Computes the derivative of 'z^l_n' with respect to the layer's biases
        'b^l_n', which turns out to be just ones.
        """
        return ArrayUtils.all_ones_array((self.layer_sizes()[layer_id],))

    def _compute_back_prop_monomer_for_target_l(self, l: int) -> np.array:
        """
        Computes the pair of partial derivs that form a repeated monomer when
        calculating delta_weights and delta_biases for interior layers.

        l: int - The monomer crosses layers. This parameter designates the 
            upstream target layer, not the source layer.
        """
        return np.einsum('nm,n -> nm',
                         self._deriv_z_wrt_a_m1(l+1), 
                         self._deriv_a_wrt_z(l))

    def compute_delta_weights_and_biases(self, labels: np.array, learning_rate: np.float32) -> list:
        """
        Computes delta_weights_and_biases for all layers, excluding the 0th.

        labels: np.array - ground truth values for output
        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        
        returns a list of tuples of matrices
        """
        f = len(self.layer_sizes()) - 1
        start_monomer = self._deriv_Cost_wrt_a_output(labels, self.outputs()) * self._deriv_a_wrt_z(f)
        delta_biases = [start_monomer]
        # delta_weights only differs from delta_biases by the final 
        # _deriv_z_wrt_weights(l) term, see below. We start by assembling 
        # delta_biases and then apply the weights term at the end. No such 
        # suffix is required for bias deltas.
        for l in range(f-1, 0, -1):
            # Prepend next monomer
            delta_biases.insert(0,
                np.dot(delta_biases[0], self._compute_back_prop_monomer_for_target_l(l).T)
            )
        # Insert an empty entry for the input (0th) layer
        delta_biases.insert(0, np.empty(()))

        delta_weights = []
        for l, vector in enumerate(delta_biases):
            delta_weights.append(
                -learning_rate * np.outer(vector, self._deriv_z_wrt_weights(l)).T
            )
            vector *= -learning_rate

        return list(zip(delta_weights, delta_biases))

class RandomUtils():
    def __init__(self, seed: int):
        self.R = Random()
        self.R.seed(seed)

    def random(self) -> np.float32:
        return self.R.random()

    def tan_random_float(self) -> np.float32: 
        return (lambda x: tan(2.*pi*(x - 0.5)))(self.R.random())

    def random_array(self, shape: tuple) -> np.array:
        ranlist = []
        for _ in range(prod(shape)):
            ranlist.append(self.tan_random_float())
        # https://opensourceoptions.com/blog/10-ways-to-initialize-a-numpy-array-how-to-create-numpy-arrays/
        return np.array(ranlist).reshape(shape)

class ArrayUtils():
    def all_zeros_array(shape: tuple) -> np.array:
        return np.zeros(shape, dtype=np.float32)

    def all_ones_array(shape: tuple) -> np.array:
        return np.ones(shape, dtype=np.float32)
class Activation():
    """
    Implementation of the classic node activation function called the sigmoid 
    function, $\\frac{1}{1 + e^{-x}}$. Despite the fact that I will inevitably 
    be coding up the derivative before long, I don't think I'm going to make 
    this a class. I don't want the overhead of a constructor.
    """
    def sigmoid(input: np.float32) -> np.float32:
        # Overflows will often show up here. There's no reason to try and catch
        # them. It's natural for the code to crash when it's wildly diverging.
        return 1./(1. + exp(- input))

    def deriv_sig(input: np.float32) -> np.float32:
        emx:np.float32 = exp(-input)
        assert emx > 0.,\
            f'deriv_sig: exp(-x) should be strictly positive, not {emx}'
        return np.float32(- emx/((1. + emx)*(1. + emx)))

class CrossEntropyImpl():
    def cost(labels: np.array, predictions: np.array) -> np.float32:
        assert len(labels.shape) == len(predictions.shape) == 1
        assert labels.shape[0] == predictions.shape[0]

        vlog = np.vectorize(log)
        return - np.sum(np.dot(labels, vlog(predictions)) + np.dot((1. - labels), vlog(1. - predictions)))

    def cost_deriv(labels: np.array, predictions: np.array) -> np.array:
        assert len(labels.shape) == len(predictions.shape) == 1
        assert labels.shape[0] == predictions.shape[0]

        return - (labels/predictions + (1. - labels)/(1. - predictions))
