import numpy as np

class Node():
    def __init__(self, init_input_weights_1D: np.array, input_layer):
        assert(input_layer.size() > 0)
        # init_input_weights_1D must be 1D for a simple node
        assert(init_input_weights_1D.shape == (input_layer.size(), 1) or \
                init_input_weights_1D.shape == (input_layer.size(), ))
            
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_weights = init_input_weights_1D

    def output(self) -> np.float32:
        return sum(map(lambda x,y: x*y, self.input_weights, self.input_layer.outputs()))


class Layer():
    def __init__(self, size: int, init_input_weights_2D: np.array, input_layer):
        assert(size > 0)
        assert(len(init_input_weights_2D.shape) == 2)
    
        # If the immediately upstream layer is a Value_layer, require that it 
        # have same size as this layer.
        if isinstance(input_layer, Value_layer):
            assert input_layer.size() == size,\
            'upstream is Value_layer -> sizes must match'
        
        assert init_input_weights_2D.shape == (input_layer.size(), size),\
        'init_input_weights_2D must be a matrix of dimension ({}, {}) not {}!'\
        .format(input_layer.size(), size, init_input_weights_2D.shape)

        self.nodes = []
        for n in range(size):
            self.nodes.append(Node(init_input_weights_2D[:,n], input_layer))

    def size(self):
        return len(self.nodes)

    def outputs(self) -> list[np.float32]:
        # inputs = input_layer.outputs()
        return [node.output() for node in self.nodes]


class ValueLayer():

    def __init__(self, global_input_values: list[np.float32]):
class Value_layer():
    def __init__(self, global_input_values: np.array):
        assert(np.linalg.matrix_rank(global_input_values) == 1)
        self.global_input_values = global_input_values

    def size(self):
        return self.global_input_values.shape[0]

    def outputs(self) -> list[np.float32]:
        return self.global_input_values

    def adjust_global_input_values(self, global_input_values: np.array):
        self.global_input_values = global_input_values

input_values = [1.0, -0.5]
layer0 = ValueLayer(input_values)
layer1 = Layer(3, layer0)
input_values = np.array([1.0, -0.5, 2.0])
layer0 = Value_layer(input_values)
layers = [layer0, layer1]

print([val for val in layers[-1].outputs()])