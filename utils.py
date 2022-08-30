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

    def identity_arrays_and_uniform_vectors(shape: tuple, 
                                            vector_val: np.float32) -> np.array:
        if len(shape) == 2:
            result = np.zeros(shape, dtype=np.float32)
            for n in range(min(*shape)):
                result[n,n] = 1.
            return result
        else:
            assert len(shape) == 1,\
                'Desired rank wasn\'t 2 or 1. Confused.'
            result = np.ones(shape, dtype=np.float32)
            return vector_val*result


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
        deriv_sigmoid_func = np.vectorize(lambda emx: emx/((1. + emx)*(1. + emx)))
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

        return (lbls/prds) / -num_examples


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
        assert len(examples.shape) == 2
        return (examples[0:num_features,...], examples[num_features:,...])


class DeltasFunnel():
    def average(weightsAndBiases: list) -> list:
        num_layers = len(weightsAndBiases)
        num_examples = weightsAndBiases[1][0].shape[1]
        result = [None] * num_layers

        for l,layerLweightsAndBiases in enumerate(weightsAndBiases):
            if l > 0:
                thisWeights, thisBiases = layerLweightsAndBiases
                weights = np.sum(thisWeights, axis=-1) / num_examples
                biases = np.sum(thisBiases, axis=-1) / num_examples
                result[l] = (weights, biases)
            else:
                result[l] = (None, None)
        return result

class DeltaWeightsAndBiases:
    """
    Defines holder class for weights and biases updates generally known as 
    delta_weights_and_biases.

    delta_weights_and_biases is a list of length num_layers, where each element
    is a list containing two numpy.arrays representing that layer's weights
    and biases respectively.
    """
    def __init__(self, layer_sizes: tuple, num_examples: int):
        self.num_layers = len(layer_sizes)
        self.contents = [(None,None)]
        for l in range(1, self.num_layers):
            this_layer_size = layer_sizes[l]
            last_layer_size = layer_sizes[l-1]
            self.contents.append([np.empty((last_layer_size, this_layer_size, num_examples), dtype=np.float32), 
                                  np.empty((this_layer_size, num_examples), dtype=np.float32)])

    @classmethod
    def ingest(self, delta_weights_and_biases: list):
        self.num_layers = len(delta_weights_and_biases)
        num_examples = delta_weights_and_biases[1][0].shape[-1]
        layer_sizes = [delta_weights_and_biases[1][0].shape[0]]
        for l in range(1, self.num_layers):
            layer_sizes.append(delta_weights_and_biases[l][1].shape[0])

        self.contents = DeltaWeightsAndBiases(layer_sizes, num_examples).contents
        for l in range(1, self.num_layers):
            for t in range(2):
                self.contents[l][t] = delta_weights_and_biases[l][t]

        return self

    @classmethod
    def copy(self):
        num_examples = self._getNumExamples
        layer_sizes = self._getLayerSizes
        num_layers = len(layer_sizes)
        result = DeltaWeightsAndBiases(layer_sizes, num_examples)
        for l in range(1, num_layers):
            for t in range(2):
                result[l, t] = self.contents[l, t]
        return result

    @classmethod
    def _getLayerSizes(self):
        layer_sizes = [self.contents[1][0].shape[0]]
        for l in range(1, self.num_layers):
            layer_sizes.append(self.contents[l][1].shape[0])
        return layer_sizes

    @classmethod
    def getNumLayers(self):
        return len(self.contents)

    def _getNumExamples(self):
        return self.contents[1,0].shape[-1]

    def __getitem__(self, indices: tuple) -> np.array:
        assert len(indices) == 2
        l, t = indices
        assert l != 0,\
            'DeltaWeightsAndBiases: no access to layer 0 weights and biases.'
        assert l < self.num_layers
        assert t < 2
        return self.contents[l][t]

    def __setitem__(self, indices: tuple, value):
        assert len(indices) == 2
        l, t = indices
        assert l != 0,\
            'DeltaWeightsAndBiases: no access to layer 0 weights and biases.'
        assert l < self.num_layers
        assert t < 2
        self.contents[l][t] = value

    @classmethod
    def average(self):
        """
        Reduces a DeltaWeightsAndBiases (DWABs) representing a batch of 
        delta_weights_and_biases down to just one by averaging them. Useful 
        for applying micro-batch weights and biases updates.

        Parent is expected to contain the batch of DWABs to be averaged. 
            Individual DWABs are represented via the examples (final) index 
            on each layer's numpy.arrays.

        Returns a reduced DeltaWeightsAndBiases without (or rather, reduced to 
        dimension 1) the extra index on each layer's weights and biases arrays.
        """
        num_examples = self.contents[1][0].shape[-1]
        num_layers = len(self.contents)
        layer_sizes = self._getLayerSizes()

        result = DeltaWeightsAndBiases(layer_sizes, 1)
        for l in range(1, num_layers):
            for t in range(2):
                result[l, t] = np.add.reduce(self.contents[l][t], axis=-1, keepdims=False) / num_examples
                assert result[l, t].shape == self.contents[l][t].shape[:-1]

        return result


        # for l in range(1, num_layers):
        #     # element assignment doesn't work on tuples so we cast the result[l] to list
        #     result[l] = list(deltas[l])
        #     for t in range(2):
        #         # print(f'average: l={l} and t={t}: deltas={deltas[l][t].shape}')
        #         result_new_shape = deltas[l][t].shape[:-1]
        #         result[l][t] = np.add.reduce(result[l][t], axis=-1, keepdims=False)
        #         # print(f'average: l={l} and t={t}: result={result[l][t].shape}')
        #         result[l][t] /= num_deltas
        #         assert result[l][t].shape == result_new_shape
        #     # And finally we cast it back to tuple
        #     result[l] = tuple(result[l])
        # return result

class PreparatoryUtils():
    def fan_out_categories_to_separate_outputs(_label: int, num_outputs: int):
        """
        Takes single categorical labels (e.g. 0,1,2) and converts them to 
        num_outputs category specific outputs.

        t: int - label for an example
        returns numpy array with only a single 1
        """
        result = np.zeros(num_outputs, dtype=np.int16)
        result[_label] = 1
        return result

    def batch_examples(block_size: int, examples: np.array) -> list:
        """
        Collects examples from the input into individual batches, with each 
        batch containing exactly block_size of them.

        block_size: int - Number of examples in each batch
        examples: numpy.array - input for set of examples
        returns a list of batches
        """
        num_features, num_examples = examples.shape
        num_blocks = num_examples//block_size
        assert num_blocks*block_size == num_examples
        blocks = [np.empty((num_features, block_size)) for _ in range(num_blocks)]

        for m in range(num_blocks*block_size):
            blocks[m//block_size][:,m%block_size] = examples[:,m]
        return blocks

    def average_of_distinct_deltas(deltas: list):
        # delta_weights_and_biases is a list of tuples of np.array's.
        # deltas is the list of those.
        num_deltas = len(deltas)
        num_layers = len(deltas[0])

        result = deltas[0]
        for d in range(1, num_deltas):
            this_deltas = deltas[d]
            for l in range(num_layers): # layer index
                if l > 0:
                    for t in range(2): # tuple index
                        result[l][t] += this_deltas[l][t]

        for l in range(num_layers):
            if l > 0:
                for t in range(2):
                    result[l][t] /= num_deltas
        return result


