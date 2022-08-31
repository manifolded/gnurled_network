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
from utils import CategoricalCrossEntropy, MeanVarianceConditioner, InstanceLabelZipper, ArrayUtils, PreparatoryUtils, DeltaWeightsAndBiases

start_time = time.process_time()
rng = np.random.default_rng(12345678)

iris = load_iris()
### Gotta randomize the dataset. The ground truths are sorted. See below.

instances = iris['data'].T
num_features = instances.shape[0]

targets = np.array(list(map(lambda x: PreparatoryUtils.fan_out_categories_to_separate_outputs(x, 3), iris['target']))).T
num_predictions = targets.shape[0]

### Conditioning instances
mvc = MeanVarianceConditioner(instances)
conditioned_instances = mvc.condition(instances)

### train/test split 
examples = InstanceLabelZipper.zipper(conditioned_instances, targets)
rng.shuffle(examples, axis=1)
training_examples = examples[...,:120]
test_examples = examples[...,120:]

learning_rate = 10.0

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, 
                      (lambda x: ArrayUtils.gen_func_array(x, rng.standard_normal)), 
                      CategoricalCrossEntropy)

training_conditioned_instances, training_labels = InstanceLabelZipper.unzipper(num_features, training_examples)
num_examples = training_examples.shape[1]
print('Starting cost: ', network.cost(training_labels, network.outputs(training_conditioned_instances)))

for e in range(num_examples):
    print(f'learning on training example {e}')
    example = np.expand_dims(training_examples[:,e], axis=-1)

    example_conditioned_instances, example_labels = InstanceLabelZipper.unzipper(num_features, example)
    deltas = network.compute_DeltaWeightsAndBiases(example_labels, example_conditioned_instances, learning_rate)
    
    delta = deltas.average()

    network.add_DeltaWeightsAndBiases(delta)
    print('post example cost: ', network.cost(training_labels, network.outputs(training_conditioned_instances)))

test_conditioned_instances, test_labels = InstanceLabelZipper.unzipper(num_features, test_examples)
print('test examples cost: ', network.cost(test_labels, network.outputs(test_conditioned_instances)))

print(time.process_time() - start_time, "seconds")

