import pytest
import numpy as np
from numpy.testing import assert_array_equal

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

def test_Given_oneOneNetwork_When_unitInput_Then_sigmoid_output():
    num_examples = 3
    unit_array = ArrayUtils.all_ones_array((1, num_examples))
    network = Network((1,1), ArrayUtils.all_ones_array, BinaryCrossEntropy)
    assert_array_equal(network.outputs(unit_array), unit_array*0.867703 ) 
    
