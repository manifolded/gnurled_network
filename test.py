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
from utils import CategoricalCrossEntropy, MeanVarianceConditioner, InstanceLabelZipper

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

### Conditioning instances
mvc = MeanVarianceConditioner(instances)
cond_instances = mvc.condition(instances)

### train/test split 
examples = InstanceLabelZipper.zipper(cond_instances, targets)
rng.shuffle(examples, axis=1)

training_examples = examples[...,:120]
test_examples = examples[...,120:]

def segment_examples(block_size: int, examples: np.array) -> list:
    num_features, num_examples = examples.shape
    num_blocks = num_examples//block_size
    assert num_blocks*block_size == num_examples
    blocks = [np.empty((num_features, block_size)) for _ in range(num_blocks)]

    for m in range(num_blocks*block_size):
        blocks[m//block_size][:,m%block_size] = examples[:,m]
    return blocks

def average_deltas(deltas: list):
    # delta_weights_and_biases is a list of tuples of np.array's.
    # deltas is a list of those!!!!
    num_deltas = len(deltas)
    num_layers = len(deltas[0])

    result = []
    this_deltas = deltas[0]
    for l in range(num_layers):
        result.append(list(this_deltas[l]))
    
    for d in range(1, num_deltas):
        this_deltas = deltas[d]
        for l in range(num_layers):
            for t in range(2):
               result[l][t] += this_deltas[l][t]

    for l in range(num_layers):
        for t in range(2):
            result[l][t] /= num_deltas

    return result

batch_size = 5
miniBatches = segment_examples(batch_size, training_examples)
num_batches = len(miniBatches)

learning_rate = 0.1

### Construct network
layer_sizes = (4,5,3)
network = nwk.Network(layer_sizes, rng.standard_normal, CategoricalCrossEntropy)

for b,batch in enumerate(miniBatches):
    batch_cond_instances, batch_labels = InstanceLabelZipper.unzipper(num_features, batch)
    print('pre batch cost: ', network.cost(batch_labels, network.outputs(batch_cond_instances)))

    delta_weights_and_biaseses = []
    for e in range(batch_size):
        print(f'learning on training example {batch_size*b+e}')

        instance, label = InstanceLabelZipper.unzipper(num_features, batch[...,e])
        prediction = network.outputs(instance)

        # Should not be implemented for bulk processing. Only one example at a time.
        delta_weights_and_biaseses.append(network.compute_delta_weights_and_biases(label, instance, learning_rate))
    
    delta_weights_and_biases = average_deltas(delta_weights_and_biaseses)
    print(delta_weights_and_biases[0])
    network.add_delta_weights_and_biases(delta_weights_and_biases)
    print('post batch cost: ', network.cost(batch_labels, network.outputs(batch_cond_instances)))

print(time.process_time() - start_time, "seconds")
