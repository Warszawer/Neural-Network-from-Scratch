## Neural-Network-from-Scratch

*Our implementation of stochastic gradient descent loops over training examples in a mini-batch. It's possible to modify the backpropagation algorithm so that it computes the gradients for all training examples in a mini-batch simultaneously. The idea is that instead of beginning with a single input vector, x, we can begin with a matrix X=[x1x2…xm] whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms, and applying the sigmoid function everywhere. We backpropagate along similar lines. Explicitly write out pseudocode for this approach to the backpropagation algorithm. Modify network.py so that it uses this fully matrix-based approach. The advantage of this approach is that it takes full advantage of modern libraries for linear algebra. As a result it can be quite a bit faster than looping over the mini-batch. (On my laptop, for example, the speedup is about a factor of two when run on MNIST classification problems like those we considered in the last chapter.) In practice, all serious libraries for backpropagation use this fully matrix-based approach or some variant.*

Well, I've implemented it. 
Here we have our classic neural network with activation file, and neural network with matrix-method and activation file either.

Executing exec_normal.py makes use of network.py, which is the original python3 file except that I've added a timer. Here's the output:

```
Epoch 0: 9078 / 10000, elapsed time: 8.82s
Epoch 1: 9258 / 10000, elapsed time: 18.31s
Epoch 2: 9319 / 10000, elapsed time: 27.38s
...
Epoch 27: 9434 / 10000, elapsed time: 234.64s
Epoch 28: 9457 / 10000, elapsed time: 242.92s
Epoch 29: 9444 / 10000, elapsed time: 251.13s
```

network_matrix.py implements the matrix-based approach. Let's execute exec_matrix.py:

```
Epoch 0: 8216 / 10000, elapsed time: 2.59s
Epoch 1: 8365 / 10000, elapsed time: 5.05s
Epoch 2: 8375 / 10000, elapsed time: 7.49s
...
Epoch 27: 9482 / 10000, elapsed time: 67.95s
Epoch 28: 9483 / 10000, elapsed time: 70.37s
Epoch 29: 9511 / 10000, elapsed time: 72.78s
```

On my computer, the matrix-based approach is 3.5 times faster.

Note that the method remains exactly the same, therefore the differences in accuracy are only due to the randomness and shouldn't be interpreted.

