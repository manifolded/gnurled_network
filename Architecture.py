import numpy as np

class Node():

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_weights = [1.0] * self.input_layer.size()

    def output(self) -> np.float32:
        return sum(map(lambda x,y:x*y, self.input_weights, self.input_layer.outputs()))


class Layer():

    def __init__(self, size: int, input_layer):
        self.nodes = [Node(input_layer)] * size

    def size(self):
        return len(self.nodes)

    def outputs(self) -> list[np.float32]:
        # inputs = input_layer.outputs()
        return [node.output() for node in self.nodes]


class ValueLayer():

    def __init__(self, global_input_values: list[np.float32]):
        self.global_input_values = global_input_values

    def size(self):
        return len(self.global_input_values)

    def outputs(self) -> list[np.float32]:
        return self.global_input_values

    def reset_global_input_values(self, global_input_values: list[np.float32]):
        self.global_input_values = global_input_values

input_values = [1.0, -0.5]
layer0 = ValueLayer(input_values)
layer1 = Layer(3, layer0)
layers = [layer0, layer1]

print([val for val in layers[-1].outputs()])