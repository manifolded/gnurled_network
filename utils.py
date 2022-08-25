import numpy as np
from statistics import stdev, mean
from math import exp, tan, pi, prod, log

class RandomUtils():
    def tan_random_float(ran_func_default_rng: callable) -> np.float32: 
        return (lambda x: tan(2.*pi*(x - 0.5)))(ran_func_default_rng.random())

    def random_array(shape: tuple, ran_func_any_dist: callable) -> np.array:
        ranlist = []
        for _ in range(prod(shape)):
            ranlist.append(ran_func_any_dist())
        # https://opensourceoptions.com/blog/10-ways-to-initialize-a-numpy-array-how-to-create-numpy-arrays/
        return np.array(ranlist).reshape(shape)


class ArrayUtils():
    def all_zeros_array(shape: tuple) -> np.array:
        return np.zeros(shape, dtype=np.float32)

    def all_ones_array(shape: tuple) -> np.array:
        return np.ones(shape, dtype=np.float32)

    def gen_func_array(shape: tuple, func: callable) -> np.array:
        vals = []
        for _ in range(prod(shape)):
            vals.append(func())
        return np.array(vals).reshape(shape)


class Activation():
    """
    Implementation of the classic node activation function called the sigmoid 
    function, $\\frac{1}{1 + e^{-x}}$. Despite the fact that I will inevitably 
    be coding up the derivative before long, I don't think I'm going to make 
    this a class. I don't want the overhead of a constructor.
    """
    def sigmoid(input: np.array) -> np.array:
        # Overflows will often show up here. There's no reason to try and catch
        # them. It's natural for the code to crash when it's wildly diverging.
        sigmoid_func = np.vectorize(lambda x: 1./(1. + exp(x * -1.)))
        return sigmoid_func(input)

    def deriv_sig(input: np.array) -> np.array:
        exp_mx_func = np.vectorize(lambda x: exp(x * -1.))

        # emx:np.float32 = exp(-input)
        # assert emx > 0.,\
        #     f'deriv_sig: exp(-x) should be strictly positive, not {emx}'
        deriv_sigmoid_func = np.vectorize(lambda emx: - emx/((1. + emx)*(1. + emx)))
        return deriv_sigmoid_func(exp_mx_func(input))


class BinaryCrossEntropy():
    def cost(labels: np.array, predictions: np.array) -> np.array:
        assert labels.shape[0] == predictions.shape[0] == 1,\
            'Binary Cross Entropy only works for 1 output.'

        # Check if arguments are rank 1, and if so harmlessly expand them.
        lbls = labels
        if len(lbls.shape) == 1:
            lbls = np.expand_dims(labels, axis=-1)
        prds = predictions
        if len(prds.shape) == 1:
            prds = np.expand_dims(predictions, axis=-1)

        vlog = np.vectorize(log)
        num_examples = lbls.shape[1]
        return (np.dot(lbls[0,:], vlog(prds[0,:])) +\
            np.dot((1. - lbls[0,:]), vlog(1. - prds[0,:]))) / -num_examples

    def cost_deriv(labels: np.array, predictions: np.array) -> np.array:
        assert labels.shape[0] == predictions.shape[0] == 1,\
            'Binary Cross Entropy only works for 1 output.'

        # Check if arguments are rank 1, and if so harmlessly expand them.
        lbls = labels
        if len(lbls.shape) == 1:
            lbls = np.expand_dims(labels, axis=-1)
        prds = predictions
        if len(prds.shape) == 1:
            prds = np.expand_dims(predictions, axis=-1)
        _, num_examples = lbls.shape

        assert len(lbls.shape) == len(prds.shape) <= 2
        assert lbls.shape == prds.shape
        return (lbls/prds + (1. - lbls)/(1. - prds)) / -num_examples


class CategoricalCrossEntropy():
    def cost(labels: np.array, predictions: np.array) -> np.array:
        assert labels.shape == predictions.shape,\
            'labels and predictions must be arrays of the same shape.'
        assert labels.shape[0] >= 2,\
            'Categorical requires >= 2 outputs.'

        # Check if arguments are rank 1, and if so harmlessly expand them.
        lbls = labels
        if len(lbls.shape) == 1:
            lbls = np.expand_dims(labels, axis=-1)
        prds = predictions
        if len(prds.shape) == 1:
            prds = np.expand_dims(predictions, axis=-1)

        vlog = np.vectorize(log)
        num_outputs, num_examples = lbls.shape
        result = 0.0
        # Using for loop over outputs nodes to restrict ourselves to only those
        # components that need to be evaluating (no off-diagonals)
        for p in range(num_outputs):        
            # Using dot product to sum over all examples efficiently
            result += np.dot(lbls[p,:], vlog(prds[p,:]))
        result /= -num_examples
        return result

    def cost_deriv(labels: np.array, predictions: np.array) -> np.array:
        assert labels.shape == predictions.shape,\
            'labels and predictions must be arrays of the same shape.'
        assert labels.shape[0] >= 2,\
            'Categorical requires >= 2 outputs.'

        # Check if arguments are rank 1, and if so harmlessly expand them.
        lbls = labels
        if len(lbls.shape) == 1:
            lbls = np.expand_dims(labels, axis=-1)
        prds = predictions
        if len(prds.shape) == 1:
            prds = np.expand_dims(predictions, axis=-1)

        assert len(lbls.shape) == len(prds.shape) <= 2
        assert lbls.shape == prds.shape
        _, num_examples = lbls.shape

        return np.sum(lbls/prds, axis=-1) / -num_examples


class MeanVarianceConditioner():
    def __init__(self, instances: np.array):
        num_features = instances.shape[0]
        self.means = [mean(instances[f,:]) for f in range(num_features)]
        self.stdevs = [stdev(instances[f,:]) for f in range(num_features)]

    def condition(self, instances: np.array):
        num_instances = instances.shape[1]
        result = np.empty(instances.shape)
        for m in range(num_instances):
            result[:,m] = (instances[:,m] - self.means)/self.stdevs
        return result


class InstanceLabelZipper():
    def zipper(instances, labels):
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

    def unzipper(num_features: int, examples: np.array):
        return (examples[0:num_features,...], examples[num_features:,...])