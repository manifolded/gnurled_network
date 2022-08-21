#!/usr/bin/env python
import numpy as np
from random import Random
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
from network import RandomUtils

start_time = time.process_time()
ru = RandomUtils(1234)

### Construct example
input_values = np.array([-8., 72.0, 0.26])
label_values = np.array([0.6, 0.2])
example = {'features': input_values,
        'labels': label_values}

### Construct network
layer_sizes = (3,5,2)
network = nwk.Network(layer_sizes, ru.random_array)

learning_rate = 0.01

### Set inputs
network.adjust_global_input_values(example['features'])
network.print_status(0, example)

for iteration in range(1, 100):
    print(f'iteration {iteration} --- all layers weights\' calcs:')
    delta_weights_and_biases = network.compute_delta_weights_and_biases(example['labels'], learning_rate)
    network.add_delta_weights_and_biases(delta_weights_and_biases)
    network.print_status(iteration, example)

print(time.process_time() - start_time, "seconds")
