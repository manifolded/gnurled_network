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

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, ru.random_array, CrossEntropyImpl)

learning_rate = 0.1

for e, example in enumerate(training):
    print(f'iteration {e}')
    network.adjust_global_input_values(example[0])
    print(CrossEntropyImpl.cost(example[1], network.outputs()))
    delta_weights_and_biases = network.compute_delta_weights_and_biases(example[1], learning_rate)
    network.add_delta_weights_and_biases(delta_weights_and_biases)
    network.print_status(e, example)

print(time.process_time() - start_time, "seconds")
