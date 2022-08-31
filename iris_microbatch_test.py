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

targets = np.array(list(map(\
    lambda x: PreparatoryUtils.fan_out_categories_to_separate_outputs(x, 3), iris['target']))).T

### Conditioning instances
mvc = MeanVarianceConditioner(instances)
conditioned_instances = mvc.condition(instances)

### train/test split 
examples = InstanceLabelZipper.zipper(conditioned_instances, targets)
rng.shuffle(examples, axis=1)

training_examples = examples[...,:120]
test_examples = examples[...,120:]

### Divide Training Examples Into Batches
batch_size = 5
miniBatches = PreparatoryUtils.batch_examples(batch_size, training_examples)
num_batches = len(miniBatches)

learning_rate = 10.0

### Construct network
layer_sizes = (4,11,3)
network = nwk.Network(layer_sizes, 
                      (lambda x: ArrayUtils.gen_func_array(x, rng.standard_normal)), 
                      CategoricalCrossEntropy)

### Train On Batches
for b, batch in enumerate(miniBatches):
    batch_conditioned_instances, batch_labels = InstanceLabelZipper.unzipper(num_features, batch)
    print('pre batch cost: ', network.cost(batch_labels, network.outputs(batch_conditioned_instances)))

    deltas = network.compute_DeltaWeightsAndBiases(batch_labels, batch_conditioned_instances, learning_rate)
    delta = deltas.average()
    network.add_DeltaWeightsAndBiases(delta)
    print('post batch cost: ', network.cost(batch_labels, network.outputs(batch_conditioned_instances)))

print(time.process_time() - start_time, "seconds")
