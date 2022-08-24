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
from utils import BinaryCrossEntropy, ArrayUtils

@pytest.fixture
def single_feature_geometric_instances(num_examples: int):
    return np.full((1,num_examples), [e**2 for e in range(num_examples)], dtype=np.float32)

def test_Given_oneOneNetwork_When_unitInput_Then_layer_0_coalesced_output():
    num_examples = 1
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    assert_array_equal(network.layers[0]._coalesced_inputs(unit_array), unit_array) 

# def test_Given_oneOneNetwork_When_unitInput_Then_output_float32():
#     num_examples = 1
#     unit_array = ArrayUtils.all_ones_array((1, num_examples))
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
    
