# Citation: Part of the code is taken from publicly avaiable implementation on
# http://neuralnetworksanddeeplearning.com/chap1.html

import random
import numpy as np
import threading
from copy import deepcopy

# The class of DNN
class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(42)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self._reset_acquired_weights_and_biases()
        self.parent_update_lock = threading.Lock()


    def _reset_acquired_weights_and_biases(self):
        """Reset the acquired weights to zeros"""
        self.acquired_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.acquired_weights = [np.zeros((y, x))
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def get_model(self):
        """Return the present model (weights and biases)"""
        self.parent_update_lock.acquire()
        weights = deepcopy(self.weights)
        biases = deepcopy(self.biases)
        self.parent_update_lock.release()
        return [weights, biases]
    

    def apply_kid_gradient(self, weight_gradient, bias_gradient):
        """Update the model (weights, biases) by adding the graients obtained from the child node"""
        self.parent_update_lock.acquire()

        self.weights = [w - wg for w, wg in zip(self.weights, weight_gradient)]
        self.biases = [b - bg for b, bg in zip(self.biases, bias_gradient)]
        self.acquired_weights = [w + wg for w, wg in zip(self.acquired_weights, weight_gradient)]
        self.acquired_biases = [b + bg for b, bg in zip(self.acquired_biases, bias_gradient)]

        self.parent_update_lock.release()


    def use_parent_model(self, weight, bias):
        """Replace own model completely by the parent model"""
        self.parent_update_lock.acquire()
        self.weights = weight
        self.biases = bias
        self._reset_acquired_weights_and_biases()
        self.parent_update_lock.release()


    def get_and_reset_acquired_gradients(self):
        """Return the acquired gradients in the model. Also reset this to zero"""
        self.parent_update_lock.acquire()
        acquired_weights = deepcopy(self.acquired_weights)
        acquired_biases = deepcopy(self.acquired_biases)
        self._reset_acquired_weights_and_biases()
        self.parent_update_lock.release()
        return [acquired_weights, acquired_biases]


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs=1, mini_batch_size=16, eta=0.01):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-eta*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb
                       for b, nb in zip(self.biases, nabla_b)]

        self.acquired_weights = [w+eta*nw
                        for w, nw in zip(self.acquired_weights, nabla_w)]
        self.acquired_biases = [b+eta*nb
                       for b, nb in zip(self.acquired_biases, nabla_b)]
        

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return sum(np.linalg.norm(x-y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))