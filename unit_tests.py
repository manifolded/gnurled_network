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
from utils import BinaryCrossEntropy, ArrayUtils, CategoricalCrossEntropy

@pytest.fixture
def single_feature_geometric_instances(num_examples: int):
    return np.full((1,num_examples), [e**2 for e in range(num_examples)], dtype=np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_coalesced_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    assert_array_equal(network.layers[0]._coalesced_inputs(unit_array), unit_array) 

# Write type sensitive unit test that complains about np.float64

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

def test_Given_fakePredictions_When_compareBinAndCatCosts_Then_costsAgree():
    eps1 = 0.38
    labels = np.full((2,1), [[1. - eps1], [0. + eps1]], dtype=np.float32)
    eps2 = 0.053
    predictions = np.full((2,1), [[1.0 - eps2], [0.0 + eps2]], dtype=np.float32)
    cat_cost = CategoricalCrossEntropy.cost(labels, predictions) 
    bin_cost = BinaryCrossEntropy.cost(labels[0], predictions[0])
    assert_almost_equal(cat_cost, bin_cost, 6)

def test_Given_fakePredictions_When_compareBinAndCatDerivs_Then_derivsAgree():
    eps1 = 0.38
    labels = np.full((2,1), [[1. - eps1], [0. + eps1]], dtype=np.float32)
    eps2 = 0.053
    predictions = np.full((2,1), [[1.0 - eps2], [0.0 + eps2]], dtype=np.float32)
    cat_deriv = CategoricalCrossEntropy.cost_deriv(labels, predictions) 
    bin_deriv = BinaryCrossEntropy.cost_deriv(labels[0], predictions[0])
    print(cat_deriv.shape, bin_deriv.shape)
    assert_almost_equal(np.sum(cat_deriv), bin_deriv, 5)

# Next up? Back-propagation...
