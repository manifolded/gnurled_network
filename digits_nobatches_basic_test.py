#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_digits
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
from utils import CategoricalCrossEntropy, MeanVarianceConditioner, InstanceLabelZipper, ArrayUtils, PreparatoryUtils

start_time = time.process_time()
rng = np.random.default_rng(12345678)

digits = load_digits()

instances = digits['data'].T
num_features = instances.shape[0]
num_categories = 10

labels = np.array(list(map(
    lambda x: PreparatoryUtils.fan_out_categories_to_separate_outputs(x, num_categories), 
    digits['target']))).T

### No conditioning on these already reasonable features

### train/test split 
examples = InstanceLabelZipper.zipper(instances, labels)
### Gotta randomize the dataset. The ground truths are sorted. See below.
rng.shuffle(examples, axis=1)

num_examples = len(examples)
num_training = round(0.8*num_examples)
training_examples = examples[...,:num_training]
test_examples = examples[...,num_training:]

learning_rate = 10.0

training_instances, training_labels = InstanceLabelZipper.unzipper(num_features, training_examples)
# print(f'digits test: training_instances.shape = ({training_instances.shape})')
# print(f'digits test: training_labels.shape = ({training_labels.shape})')

### Construct network
layer_sizes = (64,20,10)
network = nwk.Network(layer_sizes, 
                      (lambda x: ArrayUtils.gen_func_array(x, rng.standard_normal)), 
                      CategoricalCrossEntropy)

num_training_examples = training_examples.shape[1]
for e in range(num_training_examples):
    example = training_examples[...,e]
    example = np.expand_dims(example, axis=-1)
    instance, label = InstanceLabelZipper.unzipper(num_features, example)
    print(f'Example ({e}): training set cost: ', network.cost(training_labels, 
                                                 network.outputs(training_instances)))

    deltas = network.compute_DeltaWeightsAndBiases(label, instance, learning_rate)
    ave_delta = deltas.average()
    network.add_DeltaWeightsAndBiases(ave_delta)

test_instances, test_labels = InstanceLabelZipper.unzipper(num_features, test_examples)
print(f'test set cost: {network.cost(test_labels, network.outputs(test_instances))}')

print(time.process_time() - start_time, "seconds")
