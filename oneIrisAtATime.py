#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
import sys
import os
import time

# Append project root to PYTHONPATH so the modules can be imported
# This is because we are working in a Python Application rather than a Python library
sys.path.append(
#    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    os.path.dirname(os.path.realpath(__file__))
)
import network as nwk
from utils import CategoricalCrossEntropy, MeanVarianceConditioner, InstanceLabelZipper, ArrayUtils, DeltasFunnel

start_time = time.process_time()
rng = np.random.default_rng(12345678)

iris = load_iris()
### Gotta randomize the dataset. The ground truths are sorted. See below.

def fan_out_categories_to_separate_outputs(t):
    """
    Take single categorical labels (e.g. 0,1,2) and convert to 
    individual-category outputs (in this case three of them.)
    
    t: int - label for an example
    returns numpy array with only a single 1
    """
    result = np.zeros(3)
    result[t] = 1.0
    return result

instances = iris['data'].T
num_features = instances.shape[0]

targets = np.array(list(map(fan_out_categories_to_separate_outputs, iris['target']))).T
num_predictions = targets.shape[0]

### Conditioning instances
mvc = MeanVarianceConditioner(instances)
conditioned_instances = mvc.condition(instances)

### train/test split 
examples = InstanceLabelZipper.zipper(conditioned_instances, targets)
rng.shuffle(examples, axis=1)
training_examples = examples[...,:120]
test_examples = examples[...,120:]

learning_rate = 0.01

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, 
                      (lambda x: ArrayUtils.gen_func_array(x, rng.standard_normal)), 
                      CategoricalCrossEntropy)

training_conditioned_instances, training_labels = InstanceLabelZipper.unzipper(num_features, training_examples)

for e,_ in enumerate(training_examples):
    print(f'learning on training example {e}')
    example = np.expand_dims(training_examples[:,e], axis=-1)

    example_conditioned_instances, example_labels = InstanceLabelZipper.unzipper(num_features, example)
    deltas = network.compute_delta_weights_and_biases(example_labels, example_conditioned_instances, learning_rate)
    
    delta = DeltasFunnel.average(deltas)

    network.add_delta_weights_and_biases(delta)
    print('post batch cost: ', network.cost(training_labels, network.outputs(training_conditioned_instances)))

print(time.process_time() - start_time, "seconds")
