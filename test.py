#!/usr/bin/env python
import numpy as np
#from random import Random
from sklearn.datasets import load_iris
import random
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
from network import CrossEntropyImpl, MeanVarianceConditioner

start_time = time.process_time()
rng = np.random.default_rng(12345678)

iris = load_iris()
### Gotta randomize the dataset. The ground truths are sorted. See below.

def fan_labels(t):
    result = np.zeros(3)
    result[t] = 1.0
    return result

instances = iris['data'].T
num_features = instances.shape[0]

targets = np.array(list(map(fan_labels, iris['target']))).T

mvc = MeanVarianceConditioner(instances)
cond_instances = mvc.condition(instances)


def instance_label_zipper(instances, labels):
    num_features = instances.shape[0]
    num_labels = labels.shape[0]
    num_examples = instances.shape[1]

    result = np.empty((num_features+num_labels, num_examples))
    for m in range(num_examples):
        for f in range(num_features):
            result[f,m] = instances[f,m]
        for l,r in zip(range(num_labels), range(num_features, num_features+num_labels)):
            result[r,m] = labels[l,m]
    return result

def instance_label_unzipper(num_features: int, examples: np.array):
    return (examples[0:num_features,...], examples[num_features:,...])

### train/test split 
examples = instance_label_zipper(cond_instances, targets)
rng.shuffle(examples, axis=1)

training_examples = examples[...,:120]
test_examples = examples[...,120:]

training_cond_instances, training_labels = instance_label_unzipper(num_features, training_examples)

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, rng.standard_normal, CrossEntropyImpl)

training_predictions = network.outputs(training_cond_instances)
# print(training_predictions)

print(network.cost(training_labels, training_predictions))

learning_rate = 0.1

num_training_examples = training_examples.shape[1]
for e in range(0, 5): # num_training_examples):
    print(f'learning on training example {e}')
    instance, label = instance_label_unzipper(num_features, training_examples[...,e])
    prediction = network.outputs(instance)

    # print entire training set cost
    print(network.cost(training_labels, training_predictions))
    # Should not be implemented for bulk processing. Only one example at a time.
    delta_weights_and_biases = network.compute_delta_weights_and_biases(label, instance, learning_rate)
#    print(delta_weights_and_biases[0])
    network.add_delta_weights_and_biases(delta_weights_and_biases)
#    network.print_status(e, example)
#    print(network.layers[-1].input_weights[0])

print(time.process_time() - start_time, "seconds")
