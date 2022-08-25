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
from utils import BinaryCrossEntropy, ArrayUtils, CategoricalCrossEntropy, Activation

@pytest.fixture
def single_feature_geometric_instances(num_examples: int):
    return np.full((1,num_examples), [e**2 for e in range(num_examples)], dtype=np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_coalesced_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    assert_array_equal(network.layers[0]._coalesced_inputs(unit_array), unit_array) 

# --- Write type sensitive unit test that complains about np.float64 ---
# def test_Given_oneOneNetwork_When_unitInput_Then_output_float32():
#     unit_array = ArrayUtils.all_ones_array((1, 1))
#     network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
#     assert network.outputs(unit_array).dtype == np.dtype(np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_sigmoid_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    expected_value: np.float32 = 0.731059
    assert_almost_equal(network.layers[0].outputs(unit_array), unit_array*expected_value, 6) 

def test_Given_oneOneNetwork_When_unitInput_Then_sigmoid_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    expected_value: np.float32 = 0.849548
    assert_almost_equal(network.outputs(unit_array), unit_array*expected_value, 6) 

def test_Given_oneOneNetwork_When_unitInput_Then_output_agrees():
    instances = [1., 0.5]
    input_array = np.full((1,len(instances)), instances, dtype=np.float32)
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    expected_output_values = [0.849548, 0.835134]
    expected_outputs = np.full((1,2), expected_output_values, dtype=np.float32)
    assert_almost_equal(network.outputs(input_array), expected_outputs, 6) 

def test_Given_oneOneNetwork_When_multInputs_Then_binaryCrossEntropy_cost_agrees():
    input_array = ArrayUtils.all_ones_array((1, 1))
    labels = np.full((1,1), [1.], dtype=np.float32)
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
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
                      CategoricalCrossEntropy)
    deriv_a_wrt_z = network._deriv_a_wrt_z(0, inputs)
    verified_deriv_a_wrt_z = np.array([[0.209773,0.217895,0.225348],
                                       [0.232008,0.237759,0.242497]])
                                       # Computed via Mathematica
    assert_almost_equal(deriv_a_wrt_z, verified_deriv_a_wrt_z, 6)

def test_Given_1layerNetwork_When_deriv_cost_wrt_predictions_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.)), 
                      CategoricalCrossEntropy)
    preds = network.outputs(inputs)
    deriv_cost_wrt_preds = CategoricalCrossEntropy.cost_deriv(labels, preds)
    verified_deriv_cost_wrt_preds = np.array([[-0.428224,-0.392631,-0.355144],
                                              [-0.31539,-0.272938,-0.227292]])
                                              # Computed via Mathematica
    assert_almost_equal(deriv_cost_wrt_preds, verified_deriv_cost_wrt_preds, 5)

def test_Given_2layerNetwork_When_deriv_cost_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy)
    predictions = network.outputs(inputs)
    verified_deriv_cost = np.array([[-0.421902,-0.377366,-0.332367],
                                    [-0.28685,-0.240762,-0.194051]])
                                    # Computed via Mathematica
    deriv_cost_wrt_preds = CategoricalCrossEntropy.cost_deriv(labels, predictions)
    assert_almost_equal(deriv_cost_wrt_preds, verified_deriv_cost,5)

def test_Given_2layerNetwork_When_deltaWeightsAndBiases_Then_agrees():
    inputs = 1. - np.arange(0.15, 0.75, 0.1, dtype=np.float32).reshape((2,3))
    labels = 1. - np.arange(0.10, 0.70, 0.1, dtype=np.float32).reshape((2,3))
    network = Network((2,2), 
                      (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.2)), 
                      CategoricalCrossEntropy)
    predictions = network.outputs(inputs)
    verified_deriv_a_wrt_z_l1 = np.array([[0.205451,0.207295,0.209182],
                                          [0.211101,0.213042,0.214992]])
                                          # Computed via Mathematica
    deriv_a_wrt_z_l1 = network._deriv_a_wrt_z(1, inputs)
    assert_almost_equal(deriv_a_wrt_z_l1, verified_deriv_a_wrt_z_l1, 6)


# def test_Given_1_1_network_When_backProp_Then_deltasZero():
#     network = Network((1,1), (lambda x: ArrayUtils.identity_arrays_and_uniform_vectors(x, 0.)), BinaryCrossEntropy)
#     # Compute delta_weights_and_biases, the automatic approach
#     delta_weights_and_biases = network.compute_delta_weights_and_biases(labels, inputs, 1.0)
#     # Compute deltas semi-automatically
#     predictions = network.outputs(inputs)
#     deriv_cost = BinaryCrossEntropy.cost_deriv(labels, predictions)
#     layer0_outputs = Activation.sigmoid(inputs)
#     deriv_af_wrt_zf = Activation.deriv_sig(layer0_outputs)
#     # Note the down-indexing of the deriv_cost term. This auto-dim-extending 
#     # code in the Activation class is really ugly and needs to be fixed!!!
#     delta_weights_f = np.einsum('n,n,p -> pn', deriv_cost[:,0], deriv_af_wrt_zf, layer0_outputs)
#     assert_array_equal(delta_weights_and_biases[1][0], delta_weights_f)
