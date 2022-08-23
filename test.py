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
from network import RandomUtils, CrossEntropyImpl

start_time = time.process_time()
ru = RandomUtils(1234)

iris = load_iris()
# print(iris.keys())
### Gotta randomize the dataset. The ground truths are sorted.

def fan(t):
    result = np.zeros(3)
    result[t] = 1.0
    return result

instances = iris['data']
targets = map(fan, iris['target'])

examples = list(zip(instances, targets))
random.shuffle(examples, ru.random)

training = examples[:119]
test = examples[120:]

# print(f'training[0:1]: {training[0:1]}')

# training is a list of tuples (pairs) of arrays
training_instances = np.empty((4, len(training)), dtype=np.float32)
training_labels = np.empty((3, len(training)), dtype=np.float32)
for e,example in enumerate(training):
    training_instances[:,e] = example[0][:]
    training_labels[:,e] = example[1][:]

# print(f'training_instances: {training_instances}')
# print(f'training_labels: {training_labels}')

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, ru.random_array, CrossEntropyImpl)

# training_predictions = network.outputs(training_instances)
# print(training_predictions)

### print(network.cost_M(training_labels, training_predictions))

learning_rate = 0.1

for e, example in enumerate(training):
    print(f'iteration {e}')
    instance = example[0]
    label = example[1]
    prediction = network.outputs(instance)
    print(network.cost_M(label, prediction))
    delta_weights_and_biases = network.compute_delta_weights_and_biases(label, learning_rate)
    network.add_delta_weights_and_biases(delta_weights_and_biases)
#    network.print_status(e, example)

print(time.process_time() - start_time, "seconds")
