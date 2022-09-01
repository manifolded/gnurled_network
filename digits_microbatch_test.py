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
from utils import CategoricalCrossEntropy, InstanceLabelZipper, ArrayUtils, PreparatoryUtils

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

num_examples = examples.shape[1]
num_training = round(0.8*num_examples)
print(num_examples, num_training)

training_examples = examples[...,:num_training]
test_examples = examples[...,num_training:]

batch_size = 20
miniBatches = PreparatoryUtils.batch_examples(batch_size, training_examples)
num_batches = len(miniBatches)

learning_rate = 5.0

### Construct network
layer_sizes = (64,20,10)
network = nwk.Network(layer_sizes, 
                      (lambda x: ArrayUtils.gen_func_array(x, rng.standard_normal)), 
                      CategoricalCrossEntropy)

### Train
for batch in miniBatches:
    batch_instances, batch_labels = InstanceLabelZipper.unzipper(num_features, batch)
    print('pre batch cost: ', network.cost(batch_labels, network.outputs(batch_instances)))

    deltas = network.compute_DeltaWeightsAndBiases(batch_labels, batch_instances, learning_rate)
    ave_delta = deltas.average()

    network.add_DeltaWeightsAndBiases(ave_delta)
    print('post batch cost: ', network.cost(batch_labels, network.outputs(batch_instances)))

test_instances, test_labels = InstanceLabelZipper.unzipper(num_features, test_examples)
print(f'test cost: {network.cost(test_labels, network.outputs(test_instances))}')
print(time.process_time() - start_time, "seconds")
