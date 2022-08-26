# gnurled_network

Toy neural network implementation written from scratch with a nod to object 
oriented design.

## Implementation

A Network contains a list of Layers, and a Layer contains a list of Nodes. Both
Layers and Nodes keep a reference to the previous Layer so they can request its
output. Layer0 must always be a special InjectionLayer which simply provides a
way to inject the input.

### Vectorization

None. Reexpressing everything in terms of matrix algebra is definitely a goal, but I 
want to get this one working first.

## To-Do

    [x] Add biases
    [x] Add class Network
        - To help simplify assembly of a typical network
    [x] Add add_delta_weights
    [x] Add back-propagation algorithm
    [x] Convert calcs implementation to matrix mult
        - Current implementation fails to vectorize
    [ ] Figure out how to declare the type Layer on input argument
        - Why can't a class take itself as an argument in its constructor?
    [x] Implement random initialization for weights and biases
    [x] Convert this list to md checklist
    [x] Incorporate learning rate in add_delta_weights
    [x] Implement add_delta_biases()
    [x] Implement cost functions at every level
    [ ] Figure out how to store outputs so they don't need to be recomputed all the time
    [x] Remove Node() class
    [x] Evaluate cost on all examples every time
    [x] Convert forward-prop to matrix mult
    [x] Review new cost_func argument to insure it is used comprehensively
    [x] Replace existing cost functions with cost_m
    [ ] Should I have put the example index first on things like input_values?
    [x] The weights are changing. Why not the cost?
        - This turned out to be because I was failing to recompute bulk outputs before recomputing cost
    [ ] CategoricalCrossEntropy.cost axis expansion looks fishy when supplied with scalar inputs
    [ ] Over-train network by repeating trainings on the same training set
    [x] Implement mini-batch gradient descent
    [ ] Vectorize back-propagation to speed up mini-batch gradient descent
    [x] remove tuples from delta_weights_and_biases
    [x] Move all utility classes to their own module
    [ ] Fix outputs() float64 issue - see dtype unit test 
    [x] Modify _deriv_a_wrt_z() to return (n,m) shaped arrays
    [ ] Omitted biases and weights should be None, not empty
    [ ] _deriv_a_wrt_z(), why does it recompute from input_values?
    [ ] 
