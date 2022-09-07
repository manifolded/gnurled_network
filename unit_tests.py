import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import os
import sys
# Append project root to PYTHONPATH so the modules can be imported
# This is because we are working in a Python Application rather than a Python library
sys.path.append(
    os.path.dirname(os.path.realpath(__file__))
)
from network import Network
from utils import BinaryCrossEntropy, ArrayUtils, CategoricalCrossEntropy, PreparatoryUtils, DeltaWeightsAndBiases, Sigmoid, Softmax

@pytest.fixture
def single_feature_geometric_instances(num_examples: int):
    return np.full((1,num_examples), [e**2 for e in range(num_examples)], dtype=np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_coalesced_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy, Sigmoid)
    assert_array_equal(network.layers[0]._coalesced_inputs(unit_array), unit_array) 

# --- Write type sensitive unit test that complains about np.float64 ---
# def test_Given_oneOneNetwork_When_unitInput_Then_output_float32():
#     unit_array = ArrayUtils.all_ones_array((1, 1))
#     network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
#     assert network.outputs(unit_array).dtype == np.dtype(np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_sigmoid_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy, Sigmoid)
    expected_value: np.float32 = 0.731059
    assert_almost_equal(network.layers[0].outputs(unit_array), unit_array*expected_value, 6) 

def test_Given_oneOneNetwork_When_unitInput_Then_sigmoid_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy, Sigmoid)
    expected_value: np.float32 = 0.849548
    assert_almost_equal(network.outputs(unit_array), unit_array*expected_value, 6) 

def test_Given_oneOneNetwork_When_unitInput_Then_output_agrees():
    instances = [1., 0.5]
    input_array = np.full((1,len(instances)), instances, dtype=np.float32)
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy, Sigmoid)
    expected_output_values = [0.849548, 0.835134]
    expected_outputs = np.full((1,2), expected_output_values, dtype=np.float32)
    assert_almost_equal(network.outputs(input_array), expected_outputs, 6) 

def test_Given_oneOneNetwork_When_multInputs_Then_binaryCrossEntropy_cost_agrees():
    input_array = ArrayUtils.all_ones_array((1, 1))
    labels = np.full((1,1), [1.], dtype=np.float32)
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy, Sigmoid)
    expected_cost: np.float32 = 0.163051 # via Mathematica
    assert_almost_equal(network.cost(labels, network.outputs(input_array)), 
                        expected_cost, 6) 

def test_Given_poorPredictions_When_compareBinAndCatCosts_Then_costsAgree():
    eps1 = 0.38
    labels = np.full((2,1), [[1. - eps1], [0. + eps1]], dtype=np.float32)
    eps2 = 0.053
    predictions = np.full((2,1), [[1.0 - eps2], [0.0 + eps2]], dtype=np.float32)
    cat_cost = CategoricalCrossEntropy.cost(labels, predictions) 
    bin_cost = BinaryCrossEntropy.cost(labels[0], predictions[0])
    assert_almost_equal(cat_cost, bin_cost, 6)

def test_Given_poorPredictions_When_compareBinAndCatDerivs_Then_derivsAgree():
    eps1 = 0.38
    labels = np.full((2,1), [[1. - eps1], [0. + eps1]], dtype=np.float32)
    eps2 = 0.053
    predictions = np.full((2,1), [[1.0 - eps2], [0.0 + eps2]], dtype=np.float32)
    cat_deriv = CategoricalCrossEntropy.cost_deriv(labels, predictions) 
    bin_deriv = BinaryCrossEntropy.cost_deriv(labels[0], predictions[0])
    assert_almost_equal(np.sum(cat_deriv), bin_deriv, 5)

# Next up? Back-propagation...
@pytest.fixture
def balanced_pair_activations():
    eps = 0.21
    return np.full((2,), [[1.0 - eps], [0.0 + eps]], dtype=np.float32)

def test_Given_1layerNetwork_When_deriv_a_wrt_z_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    deriv_a_wrt_z = network._deriv_a_wrt_z(0, inputs)
    verified_deriv_a_wrt_z = np.array([[[0.209773,0.217895,0.225348], [0.,0.,0.]],
                                       [[0.,0.,0.], [0.232008,0.237759,0.242497]]])
                                       # Computed via Mathematica
    assert_almost_equal(deriv_a_wrt_z, verified_deriv_a_wrt_z, 6)

def test_Given_1layerNetwork_When_deriv_cost_wrt_predictions_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    preds = network.outputs(inputs)
    deriv_cost_wrt_preds = CategoricalCrossEntropy.cost_deriv(labels, preds)
    verified_deriv_cost_wrt_preds = np.array([[-0.428224,-0.392631,-0.355144],
                                              [-0.31539,-0.272938,-0.227292]])
                                              # Computed via Mathematica
    assert_almost_equal(deriv_cost_wrt_preds, verified_deriv_cost_wrt_preds, 5)


### 2-layer network offers testing of back-propagation
def test_Given_2layerNetwork_When_deriv_cost_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    predictions = network.outputs(inputs)
    verified_deriv_cost = np.array([[-0.421902,-0.377366,-0.332367],
                                    [-0.28685,-0.240762,-0.194051]])
                                    # Computed via Mathematica
    deriv_cost_wrt_preds = CategoricalCrossEntropy.cost_deriv(labels, predictions)
    assert_almost_equal(deriv_cost_wrt_preds, verified_deriv_cost, 5)

def test_Given_2layerNetwork_When_deltaWeightsAndBiases_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_deriv_a_wrt_z_l1 = np.array([[[0.205451,0.207295,0.209182], [0.,0.,0.]],
                                          [[0.,0.,0.], [0.211101,0.213042,0.214992]]])
                                          # Computed via Mathematica
    deriv_a_wrt_z_l1 = network._deriv_a_wrt_z(1, inputs)
    assert_almost_equal(deriv_a_wrt_z_l1, verified_deriv_a_wrt_z_l1, 6)

def test_Given_2layerNetwork_When_derivZwrtWeights_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_deriv_z_wrt_weights = np.array([[0.700567,0.679179,0.65701],
                                             [0.634136,0.610639,0.586618]])
                                              # Computed via Mathematica
    deriv_z_wrt_weights = network._deriv_z_wrt_weights(1, inputs)
    assert_almost_equal(deriv_z_wrt_weights, verified_deriv_z_wrt_weights, 5)

def test_Given_2layerNetwork_When_computeBiases_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_delta_biases = np.array([[0.0866802,0.0782261,0.069525],
                                      [0.0605543,0.0512924,0.0417194]])
                                      # Computed via Mathematica
    delta_wAndB = network.compute_DeltaWeightsAndBiases(labels, inputs, 1.)
    # delta_biases = delta_wAndB[-1][1]
    delta_biases = delta_wAndB[-1, 1]
    assert_almost_equal(delta_biases, verified_delta_biases, 7)

def test_Given_2layerNetwork_When_computeWeights_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_delta_weights = np.array([[[0.0607253,0.0531295,0.0456787],[0.0424223,0.0348367,0.0274101]],
                                       [[0.054967,0.0477679,0.0407846],[0.0383996,0.0313211,0.0244733]]])
                                         # Computed via Mathematica
    delta_wAndB = network.compute_DeltaWeightsAndBiases(labels, inputs, 1.)
    # delta_weights = delta_wAndB[-1][0]
    delta_weights = delta_wAndB[-1, 0]
    assert_almost_equal(delta_weights[:,:,0], verified_delta_weights[:,:,0], 7)

### 3-layer network offers testing of back-propagation with monomers
def test_Given_3layerNetwork_When_computePredictions_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_predictions = np.array([[0.713218,0.712315,0.711368],[0.71038,0.709353,0.708292]])
                                      # Computed via Mathematica
    predictions = network.outputs(inputs)
    assert_almost_equal(predictions, verified_predictions, 6)

def test_Given_3layerNetwork_When_computeBiases_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_delta_biases = np.array([[0.0176759,0.0159029,0.0140879],[0.0122278,0.01032,0.00836197]])
                                      # Computed via Mathematica
    delta_wAndB = network.compute_DeltaWeightsAndBiases(labels, inputs, 1.)
    # delta_biases = delta_wAndB[1][1]
    delta_biases = delta_wAndB[1, 1]
    assert_almost_equal(delta_biases, verified_delta_biases, 7)

def test_Given_3layerNetwork_When_computeWeights_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy,
                      Sigmoid)
    verified_delta_weights = np.array([[[0.0123831,0.0108009,0.00925588],[0.00856641,0.0070091,0.0054939]],
                                       [[0.0112089,0.00971091,0.00826419],[0.0077541,0.00630178,0.00490528]]])
                                      # Computed via Mathematica
    delta_wAndB = network.compute_DeltaWeightsAndBiases(labels, inputs, 1.)
    # delta_weights = delta_wAndB[1][0]
    delta_weights = delta_wAndB[1, 0]
    assert_almost_equal(delta_weights, verified_delta_weights, 7)

### Test Misc Funcs For Microbatching
@pytest.fixture
def toy_deltas():
    result = [(None, None), (np.zeros((1,2,3)), np.zeros((2,3))), (np.zeros((2,1,3)), np.zeros((1,3)))]
    weights1 = result[1][0]
    weights1[0,0,0] = weights1[0,1,1] = 1
    biases1 = result[1][1]
    biases1[0,0] = biases1[1,1] = 1
    weights2 = result[2][0]
    weights2[0,0,0] = weights2[1,0,1] = 1
    biases2 = result[2][1]
    biases2[0,0] = 1
    return result

@pytest.fixture
def average_delta_from_toy_deltas():
    result = [(None,None), (np.zeros((1,2)), np.zeros((2,))), (np.zeros((2,1)), np.zeros((1,)))]
    weights1 = result[1][0] 
    weights1[0,0] = weights1[0,1] = 0.333333
    biases1 = result[1][1]
    biases1[0] = biases1[1] = 0.333333
    weights2 = result[2][0]
    weights2[0,0] = weights2[1,0] = 0.333333
    biases2 = result[2][1]
    biases2[0] = 0.333333
    return result

def test_Given_deltaWeightsAndBiases_When_construct_Then_agrees(toy_deltas):
    num_examples = toy_deltas[1][0].shape[-1]
    num_layers = len(toy_deltas)
    layer_sizes = [toy_deltas[1][0].shape[0]]
    for l in range(1,num_layers):
        layer_sizes.append(toy_deltas[l][1].shape[0])

    delta = DeltaWeightsAndBiases(layer_sizes, num_examples)
    for l in range(1, num_layers):
        for t in range(2):
            delta[l, t] = toy_deltas[l][t]

    for l in range(1, num_layers):
        for t in range(2):
            assert_array_equal(delta[l, t], toy_deltas[l][t])
    

def test_Given_deltaWeightsAndBiases_When_takeAverage_Then_agrees(toy_deltas, average_delta_from_toy_deltas):
    toy_DWABs = DeltaWeightsAndBiases.ingest(toy_deltas)
    num_layers = toy_DWABs.getNumLayers()
    average_DWABs = toy_DWABs.average()

    for l in range(1, num_layers):
        for t in range(2):
            assert_almost_equal(average_DWABs[l, t], average_delta_from_toy_deltas[l][t], 6)


def test_Given_toyDWABs_When_applyToNetwork_Then_success(toy_deltas, average_delta_from_toy_deltas):
    toy_DWABs = DeltaWeightsAndBiases.ingest(toy_deltas)
    num_layers = toy_DWABs.getNumLayers()
    average_DWABs = toy_DWABs.average()

    network = Network((1,2,1), ArrayUtils.all_zeros_array, BinaryCrossEntropy, Sigmoid)
    network.add_DeltaWeightsAndBiases(average_DWABs)
    
    for l in range(1, num_layers):
        for t in range(2):
            assert_almost_equal(network.layers[l].get_WeightsAndBiases()[t], average_delta_from_toy_deltas[l][t], 6)


### Need unit tests for uneven layer sizes to test for p,n v. n,p ambiguity, 
### especially when we go to transpose defn for weights


### Softmax Tests
@pytest.fixture
def softmax_some_inputs():
    return np.full((3,3), [[0.93, 0.91, 0.], [-0.3, 0.91, 0.], [0.0, -0.91, 0.]])

@pytest.fixture
def softmax_some_expected_outputs():
    return np.full((3,3), [[0.592822, 0.462529, 0.333333],
                           [0.173278, 0.462529, 0.333333],
                           [0.23390, 0.0749416, 0.333333]])

def test_Given_softmax_When_someInputs_Then_agrees(softmax_some_inputs, softmax_some_expected_outputs):
    assert_almost_equal(Softmax.activation(softmax_some_inputs), softmax_some_expected_outputs, 6)

@pytest.fixture
def deriv_softmax_some_expected_outputs():
    return np.full((3,3,3), [[[0.241384,0.248596,0.222222], [-0.102723,-0.213933,-0.111111], [-0.138661,-0.0346627,-0.111111]],
                            [[-0.102723,-0.213933,-0.111111],[0.143252,0.248596,0.222222],[-0.0405297,-0.0346627,-0.111111]],
                            [[-0.138661,-0.0346627,-0.111111],[-0.0405297,-0.0346627,-0.111111],[0.179191,0.0693254,0.222222]]])

def test_Given_deriv_softmax_When_someInputs_Then_agrees(softmax_some_inputs, deriv_softmax_some_expected_outputs):
    assert_almost_equal(Softmax.derivative(softmax_some_inputs), deriv_softmax_some_expected_outputs, 6)


