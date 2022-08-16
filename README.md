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

- Add biases
- Add class Network
    - To help simplify assembly of a typical network
- Add add_delta_weights
- Add back-propagation algorithm
- Convert calcs implementation to matrix mult
    - Current implementation fails to vectorize
- figure out how to declare the type Layer on input argument
    - Why can't a class take itself as an argument in its constructor?

