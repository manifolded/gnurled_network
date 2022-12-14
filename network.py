import numpy as np
from utils import RandomUtils, ArrayUtils, DeltaWeightsAndBiases
from numpy.testing import assert_array_equal, assert_almost_equal
from typing import List, Tuple
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
    def __init__(
        self, 
        size: int, 
        init_input_weights_2D: np.array, 
        init_biases_1D: np.array, 
        input_layer: object,
        activation: object,
    ):
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
        self.activation = activation
        self.input_weights = init_input_weights_2D
        self.biases = init_biases_1D

    def size(self) -> int:
        return self.size

    def add_delta_weights_and_biases_at_layer(self, delta_weights_and_biases: Tuple[np.array]):
        delta_weights = delta_weights_and_biases[0]
        assert len(delta_weights.shape) == 2,\
            'delta_weights rank must be 2, not {}'\
            .format(len(delta_weights.shape))
        assert delta_weights.shape[1] == self.size,\
            'Number of nodes {} must match delta_weights\' 2nd extents {}'\
            .format(self.size, delta_weights.shape[1])
        assert delta_weights.shape[0] == self.input_layer.size,\
            'Number of nodes in previous layer {} must match delta_weight\'s '\
            +'first index extents {}'\
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

    def get_WeightsAndBiases(self) -> tuple:
        return (self.input_weights, self.biases)

    def _coalesced_inputs(self, input_values: np.array) -> np.array:
        """
        Returns an array of the 'pre-activation outputs' aka 'coalesced 
        inputs' of this layer.
        """
        result = input_values
        if self.input_layer != None:
            input_layer_outputs = self.input_layer.outputs(input_values)
            result = np.einsum('pn,pm -> nm', self.input_weights, input_layer_outputs)
            broad_biases = self.biases
            if len(broad_biases.shape) == 1 and len(result.shape) > 1:
                broad_biases = np.expand_dims(self.biases, axis=-1)
            result +=  broad_biases
        return result

    def outputs(self, input_values: np.array) -> np.array:
        # return np.vectorize(self.activation.activation)(self._coalesced_inputs(input_values))
        return self.activation.activation(self._coalesced_inputs(input_values))

class Network():
    """
    Holds the list of Layers that defines the network. Also provides convenient
    initialization and update methods.
    """
    def __init__(self, 
        layer_sizes: tuple, 
        array_generator: object, 
        cost_implementation: object,
        activations: List[object]
    ):
        if isinstance(layer_sizes, Tuple):
            assert all([size > 0 for size in layer_sizes])
        elif isinstance(layer_sizes, int):
            assert layer_sizes > 0
            layer_sizes = (layer_sizes,)
        else:
            assert False, 'Unreckonized layer_sizes specification. Try again.'

        self.cost_implementation = cost_implementation
        self.activations = activations
        assert len(activations) == len(layer_sizes)

        self.layers = []
        # Insert Layer 0 - input layer
        self.layers.append(Layer(layer_sizes[0], 
                                 None,
                                 None,
                                 None,
                                 activations[0],
                                 ))
        # Insert all the rest
        # Skip the first, already constructed, entry
        for idx, size in enumerate(layer_sizes):
            if(idx >= 1):
                self.layers.append(Layer(size, 
                                         array_generator((layer_sizes[idx - 1], size)),
                                         array_generator((size,)),
                                         self.layers[idx - 1],
                                         activations[idx],
                                         ))

    def layer_sizes(self) -> tuple:
        layer_sizes = [layer.size for layer in self.layers]
        return tuple(layer_sizes)

    def num_layers(self) -> int:
        return len(self.layer_sizes())

    def outputs(self, input_values: np.array) -> np.array:
        return self.layers[-1].outputs(input_values)

    def add_DeltaWeightsAndBiases(self, deltas: DeltaWeightsAndBiases):
        assert deltas.getNumLayers() == self.num_layers()
        for l in range(1, deltas.getNumLayers()):
            self.layers[l].add_delta_weights_and_biases_at_layer(deltas[l])

    def get_WeightsAndBiases(self) -> DeltaWeightsAndBiases:
        result = DeltaWeightsAndBiases(self.layer_sizes, 1)
        for l in range(1, self.num_layers):
            result[l] = self.layers[l].get_WeightsAndBiases()

    # ============================
    # Toolkit for Back-Propagation 
    def cost(self, labels: np.array, outputs: np.array) -> np.array:
        assert len(labels.shape) == len(outputs.shape)
        final_layer_size = self.layer_sizes()[-1]
        assert labels.shape[0] == outputs.shape[0] == final_layer_size
        if len(labels.shape) > 1:
            assert labels.shape[1] == outputs.shape[1] 

        return self.cost_implementation.cost(labels, outputs)

    def _deriv_Cost_wrt_a_output(self, labels: np.array, outputs: np.array = None) -> np.array:
        """
        Computes the derivative of the cost function with respect to the 
        network's outputs.

        labels: np.array - ground truth results expected
        outputs: np.array - the network's prediction
        """
        outs = outputs if outputs is not None else self.outputs()
        return self.cost_implementation.cost_deriv(labels, outs)

    def _deriv_a_wrt_z(self, 
            layer_id: int, 
            input_values: np.array
        ) -> np.array:
        """
        Computes the derivative of the network's outputs with respect to the 
        nodes' coalesced inputs, often denoted as 'z^l_n'.

        layer_id: int - indicates which layer's outputs to be differentiated
        z_layer_inputs: np.array - contains the PRE-activation node values for 
        layer 'l'. These may be known either as coalesced inputs or 
        pre-activation outputs. It means the same thing. Designated z^l_n to 
        distinguish them from layer activations, always designated a^l_n.
        """
        return self.activations[layer_id].derivative(
            self.layers[layer_id]._coalesced_inputs(input_values)
        )
        
    def _deriv_z_wrt_weights(self, layer_id: int, input_values: np.array) -> np.array:
        """
        Computes the derivative of 'z^l_n' in terms of the weights, 'W^l_{m n}'
        which turns out to be nothing more than the post-activation ouputs of 
        layer 'l'. 

        layer_id: int - is 'l' which indicates to which layer these quantities belong
        """
        # Note that output is strangely independent of the layer
        #    node index. Turns out it must be. 

        #### Should I add in the usual input caching check? Not for an internal layer? ####
        return self.layers[layer_id - 1].outputs(input_values)

    def _deriv_z_wrt_a_m1(self, layer_id: int) -> np.array:
        """
        Computes the dervative of 'z^l_n' in terms of the previous layer's
        post-activation node outputs 'a^(l-1)_m' which turns out to be the 
        transpose of the weights 'W^l_{m n}'

        layer_id: int - 'l', which indicates to which layer these quantities 
            belong
        """        
        return self.layers[layer_id].input_weights.T

    def _deriv_z_wrt_b(self, layer_id: int) -> np.array:
        """
        Computes the derivative of 'z^l_n' with respect to the layer's biases
        'b^l_n', which turns out to be just ones.
        """
        return ArrayUtils.all_ones_array((self.layer_sizes()[layer_id],))

    def _compute_back_prop_monomer_for_target_l(self, l: int, input_values: np.array) -> np.array:
        """
        Computes the pair of partial derivs that form a repeated monomer when
        calculating delta_weights and delta_biases for interior layers.

        l: int - The monomer crosses layers. This parameter designates the 
            upstream target layer, not the source layer.
        """
        return np.einsum('de,egm -> dgm', # when called on layer l = f-1
                         self._deriv_z_wrt_a_m1(l+1), # layer f input weights
                         self._deriv_a_wrt_z(l, input_values)) # layer f-1 sigmoid(coalesced inputs)

    def compute_DeltaWeightsAndBiases(self, labels: np.array, input_values: np.array, 
                                      learning_rate: np.float32) -> list:
        """
        Computes delta weight and bias arrays for all layers. (Except for the 
        input layer where it just returns (None, None).)

        labels: np.array - ground truth values for output
        learning_rate: np.float32 - arbitrary coefficient for delta_biases
        
        returns a DeltaWeightsAndBiases
        """
        assert len(labels.shape) == 2,\
            'labels must be rank 2, where the examples index is last.'
        assert len(input_values.shape) == 2,\
            'input_values must be rank 2, where the examples index is last.'

        num_layers = len(self.layer_sizes())
        f = num_layers - 1
        num_inputs = self.layer_sizes()[0]
        num_outputs = self.layer_sizes()[f]
        num_examples = input_values.shape[1]

        assert labels.shape == (num_outputs, num_examples)
        assert input_values.shape == (num_inputs, num_examples)

        predictions = self.outputs(input_values)
        deriv_cost_wrt_a = self._deriv_Cost_wrt_a_output(labels, predictions)
        deriv_a_wrt_z_f = self._deriv_a_wrt_z(f, input_values)
        assert predictions.shape == (num_outputs, num_examples),\
            f'predictions.shape = {predictions.shape}'
        assert deriv_cost_wrt_a.shape == (num_outputs, num_examples),\
            f'deriv_cost_wrt_a_f.shape = {deriv_cost_wrt_a.shape}'
        assert deriv_a_wrt_z_f.shape == (num_outputs, num_outputs, num_examples),\
            f'final layer: _deriv_a_wrt_z_f.shape = {deriv_a_wrt_z_f.shape}'

        result = DeltaWeightsAndBiases(self.layer_sizes(), num_examples)
        start_monomer = np.einsum('cm,cdm -> dm', deriv_cost_wrt_a, deriv_a_wrt_z_f)

        # We will fill in the bias arrays starting at the final layer, and working
        # backwards.
        result[f,1] = start_monomer
        # delta_weights (result[:,0]), only differs from delta_biases 
        # (result[:,1]) by the final _deriv_z_wrt_weights(l) term, see below. 
        # We start by assembling delta_biases and then apply the weights term 
        # at the end. No such suffix is required for bias deltas.
        for l in range(f-1, 0, -1):
            # Postpend next monomer term as we progress through the 
            # network, starting with the final layer and working backwards.
            result[l,1] = \
                np.einsum('dm,dgm -> gm', 
                          result[l+1, 1],
                          self._compute_back_prop_monomer_for_target_l(l, input_values))

        for l in range(1, num_layers):
            biases_base_at_l = result[l,1]
            if biases_base_at_l is not None:
                assert biases_base_at_l.shape == (self.layer_sizes()[l], num_examples)
                deriv_z_wrt_weights = self._deriv_z_wrt_weights(l, input_values)
                assert deriv_z_wrt_weights.shape == (self.layer_sizes()[l-1], num_examples)

                result[l, 0] = \
                    -learning_rate * np.einsum('nm,pm->pnm', biases_base_at_l, deriv_z_wrt_weights)
                # Now we are free to include the learning rate on the biases side as well.
                biases_base_at_l *= -learning_rate
        return result
