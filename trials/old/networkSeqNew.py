# Citation: Part of the code is taken from publicly avaiable implementation on
# http://neuralnetworksanddeeplearning.com/chap1.html
#check the dataprep file and modify the code. Right now the file may contain null values for the spatial neigbhborhood and flag is associated at each row to know whether it labelled
# or unlabelled data ..................we need to choose the function based on the input row.
# a batch may contain labelled and unlabelled both type of rows  
import random
import numpy as np
import threading
import timeit
import time
import sys
#import logging
from copy import deepcopy
#from sklearn.metrics import mean_squared_error
from math import sqrt


# The class of DNN
def dist(x,y):   
    z = np.sqrt(np.sum((x-y)**2))
    #print z, np.exp(-z)
    return np.exp(-z)

def mapToFloat(x):
        #x = map(float,x)
        x = x[1:-1]
        #x = tuple(x)
        x = x.replace("'", "")
        x = tuple(x.split(','))
    
        #print x
        z = []
        #print y
        for w in x:
            temp = ()
            if w:
               #print x
               if(w != ' '):
                  # print x
                   z.append(w)
        z = list(z)
       
      #  print x[0], x[1], x[2]govindlahoti22@hotmail.com
        #print x
        r = map(float,z)
        #print "length", len(r)
        return r

def mapToFloaty(y):
        #x = map(float,x)
        y = y[1:-2]
        #print y
        #x = tuple(x)
        y = y.replace("'", "")
       # y = filter(None,y)
        y = tuple(y.split(','))
        z = []
        #print y
        
        for x in y:
            temp = ()
            if x:
               #print x
               if(x != '  '):
                  # print x
                   z.append(x) 
                  
        #z.append(temp)
        z = list(z)
            
        #y = list(filter(None,y))
      #  print x[0], x[1], x[2]govindlahoti22@hotmail.com
        #print x
        r = map(float,z)
        #print y
        #y = tuple(y)
        #print y
        #y = list(y)
      #  print x[0], x[1], x[2]
        #print y
        #r = float(y)
        #print r
        return r
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
        self.log_file = open("alka" + '.log', 'a')


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
        a = np.asarray(a)
        a.shape = (len(a),1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def feedforward_learning(self,a):
        a = np.asarray(a)
        a.shape = (len(a),1)
        activations = [a] # list to store all the activations, layer by layer
        activation = a
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
           # print (np.dot(w, activation) + b)
           # print b.shape, w.shape, activation.shape
            z = np.dot(w, activation) 
            #print "before", z.shape, b.shape
            z.shape = (len(z),1)
           # print "after", z.shape, b.shape
            #z = np.add(z, b)
            z = z + b
            zs.append(z)
            #z = np.dot(w, activation)
            #print "z.shape", z.shape
            activation = sigmoid(z)
            activations.append(activation)
            #print "feedforward", activation.shape, z.shape
        return zs, activations
    

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, alpha, beta,test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        #self.log_file = open(str(alpha) + '.log', 'a')
        #sys.stdout = open("outputSeq","w")
        print "file is opened"
        n = len(training_data)
        #n=2.566666666
        self.log('length of training data {} runtime {}'.format(n, time.time()))
        #sys.stdout.close()
        for j in xrange(epochs):
            random.shuffle(training_data)
            #if(len(training_data[0]) == 6):
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            i = 0
            print "number of minibatches", len(mini_batches)
            for mini_batch in mini_batches:
                self.update_l_mini_batch(mini_batch, eta, lmbda, alpha, beta, n)
              #  print "mini batch", i
              #  i = i+1
                #if(i==1):
                #    break
                    
            #else:
             #   l=0
             #   mini_batches = [
             #       training_data[k:k+mini_batch_size]
             #       for k in xrange(0, n, mini_batch_size)]
             #   for mini_batch in mini_batches:
             #       self.update_u_mini_batch(mini_batch, eta, lmbda, alpha, beta, n)
             #       print "mini batch", l
             #       l = l+1
             #       if(l==10):
             #           break
            print "model is learnt"       
            #print "model is learnt"
            #stop1 = timeit.default_timer()
            print "evaluation started"
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n)
            else:
                print "Epoch {0} complete".format(j)   
                
            print "RMSE"
            self.evaluatePrint(test_data) 
        #print "test data evaluation for printing results" 
	#sys.stdout.close()
        #self.evaluatePrint(test_data)
        #sys.stdout.close()

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.v
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
        
    
    def update_l_mini_batch(self, mini_batch, eta, lmbda, alpha, beta, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate. L2 regularization is added"""
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#	nabla_b_ul = [np.zeros(b.shape) for b in self.biases]
      #  nabla_w_ul = [np.zeros(w.shape) for w in self.weights]
        i=0
	mini_batch_l = 1
	mini_batch_ul = 0
	lmbda1_l=0
	lmbda1_ul=0
	
            #print mini_batch
        for x, xs1, xs2, xt1, xt2, y, flag in mini_batch:
            #print i
            flag = flag[1:-2]
           # i=i+1
           # print i
            if(flag == "1" and len(mapToFloaty(y))==48 and len(mapToFloat(x)) == 276 and len(mapToFloat(xs1)) == 276 and len(mapToFloat(xs2)) == 276 and len(mapToFloat(xt1)) == 276 and len(mapToFloat(xt2)) == 276):
                #print "1 is started"
                delta_nabla_b, delta_nabla_w = self.backprop_l(x, xs1, xs2, xt1, xt2, y, alpha, beta)
            #for nb, dnb in zip(nabla_b, delta_nabla_b):
                #print "weight update", nb.shape, dnb.shape 
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                lmbda1_l = lmbda
		mini_batch_l = mini_batch_l+1
            else:
                if(flag == "0" and len(mapToFloat(x)) == 276 and len(mapToFloat(xs1)) == 276 and len(mapToFloat(xs2)) == 276 and len(mapToFloat(xt1)) == 276 and len(mapToFloat(xt2)) == 276):
                    delta_nabla_b, delta_nabla_w = self.backprop_u(x, xs1, xs2, xt1, xt2, alpha, beta)
            
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    lmbda1_l = lmbda 
		    mini_batch_ul = mini_batch_ul+1
	
                #else:
                #    print "error"
                
        #self.weights = [w-(eta/len(mini_batch))*nw
         #               for w, nw in zip(self.weights, nabla_w)]
        #self.biases = [b-(eta/len(mini_batch))*nb
         #              for b, nb in zip(self.biases, nabla_b)]
        #print "next"
        #print nabla_b
        if(mini_batch_l>1):
		mini_batch_l = mini_batch_l-1
        if(mini_batch_ul>0 and mini_batch_l==1):
		mini_batch_l = mini_batch_l-1
#	if(mini_batch_l > 1 and mini_batch_ul > 0):
#		print mini_batch_l, mini_batch_ul
	self.weights = [w-eta*(lmbda1_l/(mini_batch_l+mini_batch_ul))*w-(eta/(mini_batch_l+mini_batch_ul))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/(mini_batch_l+mini_batch_ul))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        self.acquired_weights = [w+eta*(lmbda1_l/(mini_batch_l+mini_batch_ul))*w+(eta/(mini_batch_l+mini_batch_ul))*nw
                        for w, nw in zip(self.acquired_weights, nabla_w)]
        self.acquired_biases = [b+(eta/(mini_batch_l+mini_batch_ul))*nb
                       for b, nb in zip(self.acquired_biases, nabla_b)]
        #for b in self.biases:
        #    print b
        #    print "next layer b"
        #return weights, biases
    
     # for unlabelled data
    def update_u_mini_batch(self, mini_batch, eta, lmbda, alpha, beta, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate. L2 regularization is added"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
            #print mini_batch
        i=0
	for x, xs1, xs2, xt1, xt2 in mini_batch:
            i=i+1
	    print i
            delta_nabla_b, delta_nabla_w = self.backprop_u(x, xs1, xs2, xt1, xt2, alpha, beta)
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        #self.weights = [w-(eta/len(mini_batch))*nw
         #               for w, nw in zip(self.weights, nabla_w)]
        #self.biases = [b-(eta/len(mini_batch))*nb
         #              for b, nb in zip(self.biases, nabla_b)]
        
        
        
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        self.acquired_weights = [w+(eta/len(mini_batch))*nw
                        for w, nw in zip(self.acquired_weights, nabla_w)]
        self.acquired_biases = [b+(eta/len(mini_batch))*nb
                       for b, nb in zip(self.acquired_biases, nabla_b)]
        
        
        
    def computeNeighbor(self,x,y):
        activation_nbd = x
        activations_nbd = [x] # list to store all the activations, layer by layer
        zs_nbd = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation_nbd)+b
            zs_nbd.append(z)
            activation_nbd = sigmoid(z)
            activations_nbd.append(activation_nbd)
        return zs_nbd, activations_nbd
    
    #++++++   for labelled dataset 
    def backprop_l(self, x, xs1, xs2, xt1, xt2, y, alpha, beta):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        #print "feedforward"
        x = mapToFloat(x)
        #x = map(float,x)
        
        
    
      #  x =  map(float, x.split(','))
        #x = map(float, x)
      #  print x
      #  print x
       # x = np.array(x).transpose()
       # y = np.arange(2).reshape(1, 2)
       # l = len(x)
        #print xs1
        xs1 = mapToFloat(xs1)
        #print xs2
        xs2 = mapToFloat(xs2)
        #print xt1, xt2
        xt1 = mapToFloat(xt1)
        xt2 = mapToFloat(xt2)
        
        d1 = dist(np.asarray(x),np.asarray(xs1))
        #print "dist", d1
        zs1,a1 = self.feedforward_learning(xs1)
        d2 = dist(np.asarray(x),np.asarray(xs2))
        zs2,a2 = self.feedforward_learning(xs2)
        d3 = dist(np.asarray(x),np.asarray(xt1))
        zs3,a3 = self.feedforward_learning(xt1)
        d4 = dist(np.asarray(x),np.asarray(xt2))
        zs4,a4 = self.feedforward_learning(xt2)
       # print "size of zs1", zs1[-1].shape, zs2[-2].shape, zs3[-3].shape
        
        #print y
        #if(len(y)>1):
        #    y = mapToFloat(y)
        #else:
        y = mapToFloaty(y)
        #y = map(float, y)
        y = np.asarray(y)
        #print y
        y.shape = (len(y),1)
        #y = np.array(y).transpose()
        activation = np.asarray(x)
        activation.shape = (len(activation),1)
       # print "activation", activation.shape
       # print activation
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        u=0
        for b, w in zip(self.biases, self.weights):
           # print (np.dot(w, activation) + b)
           # print b.shape, w.shape, activation.shape
            z = np.dot(w, activation) 
            #z.shape = (len(z),1)
           # print "before", z.shape
            #z = z+b
            z = np.add(z, b)
            #z = np.dot(w, activation)
           # print "z.shape", z.shape
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
           # print "sigmoid", len(activations[-1]), len(zs[-1])
            
        # backward pass
        
        #print activations[-1]
       # delta = self.cost_derivative(activations[-1], y)
        #print delta
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        #print len(y), activations[-1].shape, delta.shape
        delta11 = self.cost_derivative(activations[-1], a1[-1]) * \
            sigmoid_prime(zs1[-1])
            
        delta1 = np.dot(delta11, a1[-2].transpose())
         
        delta12 = self.cost_derivative(activations[-1], a1[-1]) * \
            sigmoid_prime(zs[-1])
        u=u+1
        #print u, len(delta12), len(activations[-2])    
        #print  u, delta.shape, delta11.shape, delta1.shape, delta12.shape, activations[-2].shape
        delta2 = np.dot(delta12, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        
        
        
        # for second spatial neighbor
        
        
        delta21 = self.cost_derivative(activations[-1], a2[-1]) * \
            sigmoid_prime(zs2[-1])
            
        delta3 = np.dot(delta21, a2[-2].transpose())
         
        delta22 = self.cost_derivative(activations[-1],a2[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta4 = np.dot(delta22, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        
        
        # for first temporal neighbor
        delta31 = self.cost_derivative(activations[-1],a3[-1]) * \
            sigmoid_prime(zs3[-1])
            
        delta5 = np.dot(delta31, a3[-2].transpose())
         
        delta32 = self.cost_derivative(activations[-1],a3[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta6 = np.dot(delta32, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
       
        
        # for second temporal neighbor
        delta41 = self.cost_derivative(activations[-1],a4[-1]) * \
            sigmoid_prime(zs4[-1])
            
        delta7 = np.dot(delta41, a4[-2].transpose())
         
        delta42 = self.cost_derivative(activations[-1],a4[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta8 = np.dot(delta42, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        nabla_b[-1] = delta + alpha*(d1*(delta12 - delta11) + d2*(delta22 - delta21)) + beta*(d3*(delta32 - delta31) + d4*(delta42 - delta41))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) + alpha*(d1*(delta2 - delta1) + d2*(delta4 - delta3)) + beta*(d3*(delta6 - delta5) + d4*(delta8 - delta7))
       # print "shape", nabla_b[-1].shape
      #  print nabla_w[-1]
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            #print "delta", delta.shape
            z = zs[-l]
            sp = sigmoid_prime(z)
            #print "sp", sp
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #print "next layer"
            #print delta(
           # nabla_b[-l] = delta
            #print delta.shape
            activations[-l-1] = np.asarray(activations[-l-1]) 
            activations[-l-1].shape = (len(activations[-l-1]),1)
            #print activations[-l-1].shape
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
            z1 = zs1[-l]
            sp1 = sigmoid_prime(z1)
            delta11  = np.dot(self.weights[-l+1].transpose(), delta11) * sp1
            #print l, delta11.shape, a1[-l-1].shape
            
            delta1 = np.dot(delta11, a1[-l-1].transpose())
         
            delta12 = np.dot(self.weights[-l+1].transpose(), delta12) * sp
            
            delta2 = np.dot(delta12, activations[-l-1].transpose())
            
            # second neighbor
            
            z2 = zs2[-l]
            sp2 = sigmoid_prime(z2)
            delta21  = np.dot(self.weights[-l+1].transpose(), delta21) * sp2
            
            delta3 = np.dot(delta21, a2[-l-1].transpose())
         
            delta22 = np.dot(self.weights[-l+1].transpose(), delta22) * sp
            
            delta4 = np.dot(delta22, activations[-l-1].transpose())
            
            # third neighborou can change the credentials now 
            z3 = zs3[-l]
            sp3 = sigmoid_prime(z3)
            delta31  = np.dot(self.weights[-l+1].transpose(), delta31) * sp3
            
            delta5 = np.dot(delta31, a3[-l-1].transpose())
         
            delta32 = np.dot(self.weights[-l+1].transpose(), delta32) * sp
            
            delta6 = np.dot(delta32, activations[-l-1].transpose())
            
            # fourth neighbor
             
            z4 = zs4[-l]
            sp4 = sigmoid_prime(z4)
            delta41  = np.dot(self.weights[-l+1].transpose(), delta41) * sp4
            
            delta7 = np.dot(delta41, a4[-l-1].transpose())
         
            delta42 = np.dot(self.weights[-l+1].transpose(), delta42) * sp
            
            delta8 = np.dot(delta42, activations[-l-1].transpose())
             
         
        #print "cost derivative", delta.shape
            nabla_b[-l] = delta + alpha*(d1*(delta12 - delta11) + d2*(delta22 - delta21)) + beta*(d3*(delta32 - delta31) + d4*(delta42 - delta41))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) + alpha*(d1*(delta2 - delta1) + d2*(delta4 - delta3)) + beta*(d3*(delta6 - delta5) + d4*(delta8 - delta7))
       # print nabla_w    ou can change the credentials now 
        return (nabla_b, nabla_w)
    
    # backpropogatin for unlabelled dataset
    def backprop_u(self, x, xs1, xs2, xt1, xt2, alpha, beta):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        #print "feedforward"
        #print x
        x = mapToFloat(x)
      #  print x
       # x = np.array(x).transpose()
       # y = np.arange(2).reshape(1, 2)
       # l = len(x)
        xs1 = mapToFloat(xs1)
        xs2 = mapToFloat(xs2)
        xt1 = mapToFloat(xt1)
        xt2 = mapToFloat(xt2)
        
        d1 = dist(np.asarray(x),np.asarray(xs1))
        #print "dist", d1
        zs1,a1 = self.feedforward_learning(xs1)
        d2 = dist(np.asarray(x),np.asarray(xs2))
        zs2,a2 = self.feedforward_learning(xs2)
        d3 = dist(np.asarray(x),np.asarray(xt1))
        zs3,a3 = self.feedforward_learning(xt1)
        d4 = dist(np.asarray(x),np.asarray(xt2))
        zs4,a4 = self.feedforward_learning(xt2)
         # y = np.array(y).transpose()
        activation = np.asarray(x)
        activation.shape = (len(activation),1)
       # print "activation", activation.shape
       # print activation
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
           # print (np.dot(w, activation) + b)
           # print b.shape, w.shape, activation.shape
            z = np.dot(w, activation) 
           # print "before", z.shape
            z = np.add(z, b)
            #z = np.dot(w, activation)
           # print "z.shape", z.shape
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
           # print "sigmoid", len(activations[-1]), len(zs[-1])
            
        # backward pass
        
        #print activations[-1]
       # delta = self.cost_derivative(activations[-1], y)
        #print delta
        delta11 = self.cost_derivative(activations[-1],a1[-1]) * \
            sigmoid_prime(zs1[-1])
            
        delta1 = np.dot(delta11, a1[-2].transpose())
         
        delta12 = self.cost_derivative(activations[-1],a1[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta2 = np.dot(delta12, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        
        
        
        # for second spatial neighbor
        
        
        delta21 = self.cost_derivative(activations[-1],a2[-1]) * \
            sigmoid_prime(zs2[-1])
            
        delta3 = np.dot(delta21, a2[-2].transpose())
         
        delta22 = self.cost_derivative(activations[-1],a2[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta4 = np.dot(delta22, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        
        
        # for first temporal neighbor
        delta31 = self.cost_derivative(activations[-1],a3[-1]) * \
            sigmoid_prime(zs3[-1])
            
        delta5 = np.dot(delta31, a3[-2].transpose())
         
        delta32 = self.cost_derivative(activations[-1],a3[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta6 = np.dot(delta32, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
       
        
        # for second temporal neighbor
        delta41 = self.cost_derivative(activations[-1],a4[-1]) * \
            sigmoid_prime(zs4[-1])
            
        delta7 = np.dot(delta41, a4[-2].transpose())
         
        delta42 = self.cost_derivative(activations[-1],a4[-1]) * \
            sigmoid_prime(zs[-1])
            
        delta8 = np.dot(delta42, activations[-2].transpose())
         
        #print "cost derivative", delta.shape
        nabla_b[-1] = alpha*(d1*(delta12 - delta11) + d2*(delta22 - delta21)) + beta*(d3*(delta32 - delta31) + d4*(delta42 - delta41))
        nabla_w[-1] = alpha*(d1*(delta2 - delta1) + d2*(delta4 - delta3)) + beta*(d3*(delta6 - delta5) + d4*(delta8 - delta7))
        #print delta
      #  print nabla_w[-1]
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            #print "delta", delta.shape
            z = zs[-l]
            sp = sigmoid_prime(z)
            #print "sp", sp
            
            #print "next layer"
            #print delta(
           # nabla_b[-l] = delta
            #print delta.shape
            activations[-l-1] = np.asarray(activations[-l-1]) 
            activations[-l-1].shape = (len(activations[-l-1]),1)
            #print activations[-l-1].shape
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
            z1 = zs1[-l]
            sp1 = sigmoid_prime(z1)
            delta11  = np.dot(self.weights[-l+1].transpose(), delta11) * sp1
            
            delta1 = np.dot(delta11, a1[-l-1].transpose())
         
            delta12 = np.dot(self.weights[-l+1].transpose(), delta12) * sp
            
            delta2 = np.dot(delta12, activations[-l-1].transpose())
            
            # second neighbor
            
            z2 = zs2[-l]
            sp2 = sigmoid_prime(z2)
            delta21  = np.dot(self.weights[-l+1].transpose(), delta21) * sp2
            
            delta3 = np.dot(delta21, a2[-l-1].transpose())
         
            delta22 = np.dot(self.weights[-l+1].transpose(), delta22) * sp
            
            delta4 = np.dot(delta22, activations[-l-1].transpose())
            
            # third neighbor
            z3 = zs3[-l]
            sp3 = sigmoid_prime(z3)
            delta31  = np.dot(self.weights[-l+1].transpose(), delta31) * sp3
            
            delta5 = np.dot(delta31, a3[-l-1].transpose())
         
            delta32 = np.dot(self.weights[-l+1].transpose(), delta32) * sp
            
            delta6 = np.dot(delta32, activations[-l-1].transpose())
            
            # fourth neighbor
             
            z4 = zs4[-l]
            sp4 = sigmoid_prime(z4)
            delta41  = np.dot(self.weights[-l+1].transpose(), delta41) * sp4
            
            delta7 = np.dot(delta41, a4[-l-1].transpose())
         
            delta42 = np.dot(self.weights[-l+1].transpose(), delta42) * sp
            
            delta8 = np.dot(delta42, activations[-l-1].transpose())
             
         
        #print "cost derivative", delta.shape
            nabla_b[-l] = alpha*(d1*(delta12 - delta11) + d2*(delta22 - delta21)) + beta*(d3*(delta32 - delta31) + d4*(delta42 - delta41))
            nabla_w[-l] = alpha*(d1*(delta2 - delta1) + d2*(delta4 - delta3)) + beta*(d3*(delta6 - delta5) + d4*(delta8 - delta7))
       # print nabla_w    
        return (nabla_b, nabla_w)
    
        
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
        #if(len(test_data)[0]==6 || len(test_data)[0] == 2)
        if(len(test_data[0]) == 7):
            
            test_results = [(self.feedforward(mapToFloat(x)), mapToFloaty(y))
                        for (x, xs1, xs2, xt1, xt2, y,flag) in test_data]
        else:
            if(len(test_data[0])==2):
                test_results = [(self.feedforward(mapToFloat(x)), mapToFloaty(y))
                        for (x, y) in test_data]
            else:
                test_results = [(self.feedforward(mapToFloat(x)), 0.0)
                        for (x, xs1,xs2,xt1,xt2) in test_data]    
        r=0
        n = 0
        for x,y in test_results:
            if(len(y)==48):
                r = r+ np.linalg.norm(x-y)
                n=n+1
        return r/n    
        #return sum(np.linalg.norm(x-y) for (x, y) in test_results) # L2 Norm

    def evaluatePrint(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #if(len(test_data)[0]==6 || len(test_data)[0] == 2)
        if(len(test_data[0]) == 7):

            test_results = [(self.feedforward(mapToFloat(x)), mapToFloaty(y))
                        for (x, xs1, xs2, xt1, xt2, y,flag) in test_data]
        else:
            if(len(test_data[0])==2):
                test_results = [(self.feedforward(mapToFloat(x)), mapToFloaty(y))
                        for (x, y) in test_data]
            else:
                test_results = [(self.feedforward(mapToFloat(x)), 0.0)
                        for (x, xs1,xs2,xt1,xt2) in test_data]
        r=0
        n = 0
	count =0
        #f1 = open("result.txt",'w')
        for x,y in test_results:
	    if(count < 6):	
	    	print x[0],y[0]
		count = count+1
            if(len(y)==48):
                #r = r+ sqrt(mean_squared_error(y, x))
		r = r + (y[0]- x[0])*(y[0]-x[0])
                n=n+1
        print r/n
        #return sum(np.linalg.norm(x-y) for (x, y) in test_results) # L2 Norm
    


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def log(self, s):
                """Write s to the log file of this node"""
                self.log_file.write(s +'\n')
                self.log_file.flush()

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
