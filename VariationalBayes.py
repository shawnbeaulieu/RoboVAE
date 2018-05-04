#!/usr/bin/env python
# roboVAE.py
# By Shawn Beaulieu
# July 21st, 2017

"""
For use in Python 2.7

If python 2 and 3 are both installed on your system, use
"source activate py2" prior to running this code. Python 2.7 is required
for running Pyrosim. Later versions of Python are currently incompatible.

Implementation of a conditional variational autoencoder (VAE) in tensorflow.
The input data are the phenotypic weight matrices for robots evolved
using HyperNEAT. Inspiration for this code taken from the following resources:

THEORY:

(1) "Auto-encoding Variational Bayes": https://arxiv.org/pdf/1312.6114.pdf
(2) "Tutorial on VAEs": https://arxiv.org/pdf/1606.05908.pdf
(3) "Variational Inference: Review for Statisticians": https://arxiv.org/pdf/1601.00670.pdf
(4) "Thesis: Uncertainty in Deep Learning" by Yarin Gal: http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf

CODE:

(1) http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
(2) https://jmetzen.github.io/2015-11-27/vae.html
(3) Tensorflow tutorials via Tensorflow

Summary:

The purpose of a variational autoencoder (VAE) is to perform inference on  the input data, X,
so as to identify the latent features, Z, on which X is predicated (e.g. what are the (approximately) 
'irreducible' features that causally produce X?). This is useful for generating synthetic data that 
closely resembles the input data on which the network (encoder) was trained. Training consists of two 
parts: (i) using the compressed latent code, try to generate an instance that closely resembles the 
training data; (ii) backpropagate errors in reconstruction so as to better reproduce the input.
Variational inference replaces the Bayesian paradigm of marginalizing over latent variables to
perform prediction or inference with the calculation of derivatives (optimization) on the prior KL
and log likelihood terms below. These derivatives are taken with respect to the parameters of the 
two component neural networks and allow for optimization over distributions rather than point estimates
as in traditional DL models. 

The architecture is composed of two networks: an encoder and a decoder, both of which are neural networks. 
A technique called reparameterization is used in sampling the latent representation as sampling methods 
aren't amenable to backpropagation. In practice, this means that we first generate samplesfrom a standard 
normal Gaussian distribution e ~ N(0,I), which is then used to modify the latent space as follows:

z = mean + var*e, where the multiplication here is a Hadamard product between the variance and the sampled value,
epsilon.

ENCODER: q(z | x) parameterized by phi
DECODER: p(x | z) parameterized by theta

Where phi and theta refer to the parameters of the respective neural networks.

Formally, we are maximizing the expected log probability of the data given the latent representation with 
respect to the latent space, and a penalty term (KL divergence) that constrains our approximation 
of the posterior to be close to our initial prior, N(0,I) over the latent space. The motivation for this is 
to prevent the encoder from assigning a unique point in the latent space to each input instance. We want the 
encoder to be frugal in its use of the latent space for compact representation. Heuristically, this acts
as a naive Occam's Razor. See Hinton's paper on minimum description length for theoretical justification
for VI's ELBO.

All of this has the effect of maximizing the lower bound on the log probability of the data, p(X).

For more information, see the aforementioned links covering the theory of VAEs and variational inference more broadly.

"""


import os
import functools
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
from functional import compose, partial

def Xavier(name, shape):
    """
    To guard against both vanishing and exploding gradients. The variance
    of the distribution from which we draw random samples for weights
    is a function of the number of input neurons for a given layer
    (and for the case of Bengio initialization, the output neurons as well) 

    INPUTS:
    name: string containing exact name of layer being initialized
    shape: dimensions of the weight matrix: (e.g. (input,output)) 

    """
    return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True))

def Recursion(*args):
    """
    Creates a sequence of nested functions.
    - *args: a variable-length input to the function
    - functools.reduce: applies compose() to each argument in args
    - compose: strings together functions, defining a recursive function
    - partial: the returned function, which accepts a value, X, that is then
      passed through the recursive function

    >> Encoder = Recursion([f,g,z])
    >> Encoder(x) = f(g(z(x)))

    """
    
    return(partial(functools.reduce, compose)(*args))

def Convolution(inputs, weights, biases, activation=tf.nn.relu, maxpool=False):
    """

    Apply convolution operation to a given layer.
    """
    # Stride length of 1 for convolution. Padding = 'SAME' preserves original dimensionality
    convolution = tf.add(tf.nn.conv3d(inputs, weights, strides=[1,1,1,1,1], padding='SAME'), biases)
    output = activation(convolution)
    if maxpool:
        # Stride length of 2 for maxpool layer. By eq. OUT = ceil(float(IN)/STRIDE),
        # the dimensionality, per dimension, is halved.
        #output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        output = tf.nn.max_pool3d(output, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    return(output)

class VAE():

    HYPERPARAMS = {
        "batch_size": 100,
        "regularizer": 1E-5,
        "learning_rate": 1E-3,
        "activation": tf.nn.elu,
        "dropout": True,
        "dropout_rate": 0.90,
        "num_classes": 0
    }

    def __init__(self, blueprint, hyperparameters={}, convolutions=0, meta_graph=None, new_graph=True):
        """ 
        INPUTS:

        blueprint: a dictionary containing the size (number of neurons) for each layer
        activation: desired activation function for the network
        learning_rate: the magnitude with which changes to the network's weights are made
        batch_size: SGD is used for speed of convergence. Lower variance results from larger batches

        >> blueprint = [169, 500, 500, 20]
        # Where (169=input_dim, 500=first hidden layer, 500=second hl, 20=z_dim, ... )
        # VAEs are generally symmetric

        """
       
        self.blueprint = np.array(blueprint)
        self.convolutions = convolutions
        self.layer_names = range(len(blueprint))
        self.simulation_loss = None
        # Take dictionary of hyperparameters and create self.x for x in dictionary:
        # Very cool update
        self.__dict__.update(VAE.HYPERPARAMS, **hyperparameters)
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=[None, blueprint[0]], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, blueprint[-1]], name='z')
        if self.num_classes != 0:
            self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name='labels')

        # self.x is initialized as a placeholder of dimensions (? x input_features)
        # This is so that the batch size can vary dynamically, rather than be hardcoded:

        # ====================== BUILD NEW GRAPH ======================
        # Initialize tensorflow variables. Computational graph doesn't exist
        # until the variables are initialized. Said variables declared both above
        # (explicitly in the case of self.x) and in the previously called functions:

        # For the case of conditional variational autoencoding:

        # Build the network (required for both new and saved graphs)

        if new_graph:
            assert len(self.blueprint) > 2, \
                "Build Error: check layer specifications. Min length=2"

            # Create variable "new_graph" so that the optimizing function knows whether
            # to create a new optimizer or load parameters from saved model
            self.new_graph = True
            self.Parameterize()
            self.Compose_Network()
            self.Optimize()

            self.saver = tf.train.Saver()
            build = tf.global_variables_initializer()
            self.sess.run(build)

            # Save the relevant parameters, so that when loading saved graph we have all of
            # the necessary information:

        # ====================== LOAD SAVED GRAPH ======================

        # 'meta-graph' should be path to desired graph from current location.
        # e.g. ./savedVAEs/{datetime}_VAE_compression
        else:

            self.new_graph = False
            # Initialize weights and biases
            # Grab path to saved graph:
            path2meta = os.path.abspath(meta_graph)

            # Load saved graph
            saver = tf.train.import_meta_graph(path2meta)
            saver.restore(self.sess, tf.train.latest_checkpoint('./'))
            self.Restore_Parameters()
            self.Compose_Network()
            self.Optimize()

        # Save the trained network

    def Restore_Parameters(self):

        """
        In primitive state. Massively suboptimal. Grabs saved parameters
        from previously trained graph and stores them in dictionaries.

        """

        # ====================== TODO:CLEAN UP; THIS IS UGLY  ======================

        # Initialize dictionaries for weights and biases
        self.weights = {'encoder': {}, 'decoder': {}}
        self.biases = {'encoder': {}, 'decoder': {}}

        params = tf.GraphKeys.TRAINABLE_VARIABLES

        if not isinstance(self.convolutions, int):

            encoder_weights = tf.get_collection(params, "convolutions/encoder_weights")
            encoder_biases = tf.get_collection(params, "convolutions/encoder_biases")
        else:
            encoder_weights = tf.get_collection(params, "encoder_weights")
            encoder_biases = tf.get_collection(params, "encoder_biases")

        decoder_weights = tf.get_collection(params, "decoder_weights")
        decoder_biases = tf.get_collection(params, "decoder_biases")

        # Place loaded parameters into correct location in corresponding dictionary:

        weight_names = [layer.name.split("/")[-1].split(":")[0] for layer in encoder_weights]
        for theta in range(len(encoder_weights)):
            self.weights['encoder'].update({weight_names[theta]: encoder_weights[theta]})

        bias_names = [layer.name.split("/")[-1].split(":")[0] for layer in encoder_biases]
        for theta in range(len(encoder_biases)):
            self.biases['encoder'].update({bias_names[theta]: encoder_biases[theta]})

        weight_names = [layer.name.split("/")[-1].split(":")[0] for layer in decoder_weights]
        for theta in range(len(decoder_weights)):
            self.weights['decoder'].update({weight_names[theta]: decoder_weights[theta]})

        bias_names = [layer.name.split("/")[-1].split(":")[0] for layer in decoder_biases]
        for theta in range(len(decoder_biases)):
            self.biases['decoder'].update({bias_names[theta]: decoder_biases[theta]})


    def Parameterize(self):

        """
        Creates the weights and biases for the network using
        Xavier initialization for the weights of each layer, and
        initialization to 0 for all biases. Stores said parameters in
        dictionaries

        """
        self.weights = {'encoder':{}, 'decoder':{}}
        self.biases = {'encoder':{}, 'decoder':{}}

        # ====================== CONVOLUTIONAL ENCODER PARAMETERIZATION ======================
        
        # Convolutions are applied ONLY to the encoder network. The idea being that spatial
        # patterns and regularities should be detected and preserved in the compressed
        # representation. Asymmetry in power of encoder and decoder necessitates weakening
        # the decoder relative to the encoder to ensure good compression.
        # convolutions = [(height, width, input_channels, num_filters)...] = [(4,4,1,64)...]

        if not isinstance(self.convolutions, int):
            with tf.variable_scope("convolutions"):
                self.Build_Convolutions()

        # ====================== NORMAL ENCODER PARAMETERIZATION ======================

        elif isinstance(self.convolutions, int):

            # input2output is a list of tuples specifying input size, output size and the "names" of each
            # for variable declaration in tensorflow:
            input2output = zip(self.blueprint[:-1], self.blueprint[1:], self.layer_names[:-1], self.layer_names[1:])
            self.encoder_info = input2output
            # To keep ordered key names, append to list and save:
            e_weight_keys = []
            e_bias_keys = []
            for IO in input2output:
                if IO == input2output[-1]:
                    for theta in ('_mean', '_log_var'):
                        source = IO[-2]
                        layer_name = '{0}to{1}{2}'.format(source, 'z', theta)
                        weight_name = 'weights_{}'.format(layer_name)
                        bias_name = 'biases_{}'.format(layer_name)
                        e_weight_keys.append(weight_name)
                        e_bias_keys.append(bias_name)
                        with tf.variable_scope('encoder_weights'):
                            self.weights['encoder'][weight_name] = \
                                Xavier(weight_name, (IO[0], IO[1]))
                        with tf.variable_scope('encoder_biases'):
                            self.biases['encoder'][bias_name] = \
                                tf.get_variable(name=bias_name, dtype=tf.float32, initializer=tf.zeros([IO[1]]))
                                #tf.Variable(tf.zeros([IO[1]], dtype=tf.float64), name=bias_name)
                else:
                    # To account for the concatenated label vector, we add self.num_classes dimensions to the input
                    if IO == input2output[0]:
                        input_dim = IO[0] + self.num_classes
                    else:
                        input_dim = IO[0]
                    source = IO[-2]
                    target = IO[-1]
                    layer_name = '{0}TO{1}'.format(source,target)
                    weight_name = 'weights_{}'.format(layer_name)
                    bias_name = 'biases_{}'.format(layer_name)
                    e_weight_keys.append(weight_name)
                    e_bias_keys.append(bias_name)
                    with tf.variable_scope('encoder_weights'):
                        self.weights['encoder'][weight_name] = \
                            Xavier(weight_name, (input_dim, IO[1]))
                    with tf.variable_scope('encoder_biases'):
                        self.biases['encoder'][bias_name] = \
                            tf.get_variable(name=bias_name, dtype=tf.float32,initializer=tf.zeros([IO[1]]))
        
            self.e_weight_keys = e_weight_keys
            self.e_bias_keys = e_bias_keys
        # ====================== DECODER PARAMETERIZATION ======================
        
        # Shift names by length of layer_names variable: [0,1,2,3] => [3,4,5,6]:
        decoder_names = np.add(self.layer_names, self.layer_names[-1])
      
        # input2output is a list (sizeX, sizeY, nameX, nameY) using blueprint
        # specified at initialization, (e.g. [169, 500, 500, 20]). Is like the encoder
        # only in reverse:

        input2output = \
            zip(reversed(self.blueprint[1:]), reversed(self.blueprint[:-1]), decoder_names[:-1], decoder_names[1:]) 
        self.decoder_info = input2output
        d_weight_keys = []
        d_bias_keys = []
        for IO in input2output:
            # To account for the concatenated label vector:
            if IO == input2output[0]:
                input_dim = IO[0] + self.num_classes
            else:
                input_dim = IO[0]
            source = IO[-2]
            target = IO[-1]
            layer_name = '{0}TO{1}'.format(source,target)
            weight_name = 'weights_{}'.format(layer_name)
            bias_name = 'biases_{}'.format(layer_name)
            d_weight_keys.append(weight_name)
            d_bias_keys.append(bias_name)
            with tf.variable_scope('decoder_weights'):
                self.weights['decoder'][weight_name] = \
                    Xavier(weight_name, (input_dim, IO[1]))
            with tf.variable_scope('decoder_biases'):
                self.biases['decoder'][bias_name] = \
                    tf.get_variable(name=bias_name, dtype=tf.float32, initializer=tf.zeros([IO[1]]))

        self.d_weight_keys = d_weight_keys
        self.d_bias_keys = d_bias_keys

    def Build_Convolutions(self):
        """

        Creates dictionaries of weights and biases for the convolutional
        version of VAE.

        """

        # Layer names will just be index by index, e.g. 0TO1, 1TO2, etc.

        e_weight_keys = []
        e_bias_keys = []
        layer_names = range(len(self.convolutions)+1)
        for layer in self.convolutions:
            layer_name = '{0}TO{1}'.format(layer_names[0], layer_names[1])
            weight_name = 'weights_{}'.format(layer_name)
            print(weight_name)
            bias_name = 'biases_{}'.format(layer_name)
            e_weight_keys.append(weight_name)
            e_bias_keys.append(bias_name)
            # Discard first "name", so that next iteration is indexed properly.
            layer_names.pop(0)
            with tf.variable_scope('encoder_weights'):
                self.weights['encoder'][weight_name] = Xavier(weight_name, layer)
            with tf.variable_scope('encoder_biases'):
                self.biases['encoder'][bias_name] = \
                    tf.get_variable(name=bias_name, dtype=tf.float32, initializer=tf.zeros([layer[-1]]))

        # Hard code the final encoder layer, where the last convolutional layer
        # is flattened and feeds into the latent layer of size Z (self.blueprint[-1])

        final_convolution = self.convolutions[-1]

        # Dimensions of final convolutional layer: height * width * num_channels

        # Final convolution innervates two layers, corresponding to mean and variance of
        # the latent distribution.
        final_dim = self.convolutions[-1][-1]
        for theta in ('_mean', '_log_var'):
            source = layer_names[0]
            layer_name = '{0}TO{1}{2}'.format(source,'z',theta)
            weight_name = 'weights_{}'.format(layer_name)
            bias_name = 'biases_{}'.format(layer_name)
            e_weight_keys.append(weight_name)
            e_bias_keys.append(bias_name)
            with tf.variable_scope('encoder_weights'):
                self.weights['encoder'][weight_name] = \
                    Xavier(weight_name, (3*3*3*final_dim, self.blueprint[-1]))
            # Output dimension is 4x4x64 because we halve (rounded) dimensions
            # for every convolution (following pool): 14x14 => 7x7:8x8 (padded) => 4x4 
            # Because stride length = 1 for the convolution, dimensionality is independent
            # of window shape.
            with tf.variable_scope('encoder_biases'):
                self.biases['encoder'][bias_name] = \
                    tf.get_variable(name=bias_name, dtype=tf.float32, initializer=tf.zeros([self.blueprint[-1]]))

        self.e_weight_keys = e_weight_keys
        self.e_bias_keys = e_bias_keys

    def Compose_Network(self):
        """

        Steps through the network, first calling the Encode() function and obtaining
        the parameters for the latent representation (mean and variance). Next, random noise
        is sampled and then applied to the generated parameters of the compressed input via
        reparameterization. Lastly, the input data is reconstructed by Decode().

        """
        # Encode input (probabilistically):
        self.z_mean, self.z_log_var = self.Encode()

        # Sample from N(0,I):
        epsilon = tf.random_normal(shape=tf.shape(self.z_mean), mean=0.0, stddev=1, dtype=tf.float32)

        # Reparameterization:
        self.z = self.z_mean + epsilon*tf.exp(self.z_log_var)

        # Decode latent representation (probabilistically):

        #self.output_mean = self.Decode()

        # Reconstruct reconstruction:
        #if self.dropout:
        #    self.output_mean = self.Decode()
        #    self.first_KL = self.KL_Divergence()
        #    self.original = self.x
        #    self.x = self.output_mean
        #    self.z_mean, self.z_log_var = self.Encode()
        #    epsilon = tf.random_normal(shape=tf.shape(self.z_mean), mean=0.0, stddev=1, dtype=tf.float32)
        #    self.z = self.z_mean + epsilon*tf.exp(self.z_log_var)      
        #    self.output2_mean = self.Decode()
        #else:
        #    self.output2_mean = self.Decode()
        self.output_mean = self.Decode()

    def Encode(self):
        """
        Passes the input data, self.x, through the first two hidden layers and into
        the latent layer, which encodes for the mean and variance of the variational 
        distribution, Q, from which we layer sample in the decoding phase.

        """

        weights = self.weights['encoder']
        biases = self.biases['encoder']
        
        # Create tensorflow graph through which we'll pass phenotypic data:
        # Just a series of geometric transformations: X => h1 => h2 => z
    
        # ====================== TODO: USE RECURSION(); NO LOOPS  ======================
       
        # Concatenate the label vector onto the input features:
        
        if self.num_classes != 0:
            self.inputs = tf.concat(axis=1, values=[self.x, self.labels])
        else:
            self.inputs = self.x

        # If network is convolutional:
        if not isinstance(self.convolutions, int):
            z_log_var, z_mean = self.Convolutional_Encoding()

        elif isinstance(self.convolutions, int):
            # Manually code the forward pass for z_mean and z_log_var. Loop over
            # (arbitrarily many) layers with shared structure:
            # Iteratively update "encoding":
            encoding = self.inputs
            # TODO: FIX THIS. ERROR WHEN RUNNING SAVED MODEL. BANDAID SOLUTION:
            params = zip(sorted(weights.keys()), sorted(biases.keys())) 
            for p in params[:-2]: # exclude the last two "layers"
                # Variable names are: "biases_0to1", "weights_0to1", etc...
                encoding = tf.add(tf.matmul(encoding, weights[p[0]]), biases[p[1]])
                if self.dropout:
                    # Stochastic dropout. Use self.dropout_rate for static dropout
                    rate = self.dropout_rate
                    #rate = np.clip(np.random.normal(self.dropout_rate, 1), 0.1, 0.9)
                    encoding = tf.layers.dropout(inputs=self.activation(encoding), rate=rate)
                else:      
                    encoding = self.activation(encoding)

            z_log_var = tf.add(tf.matmul(encoding, weights[params[-2][0]]), biases[params[-2][1]])
            z_mean = tf.add(tf.matmul(encoding, weights[params[-1][0]]), biases[params[-1][1]])
        
        # Given that this is variational inference, which is a Bayesian inference method, the output of the
        # encoder is necessarily probabilistic. Hence, we return the parameters for a Gaussian distribution
        # corresponding to our approximation of the true posterior, q(z | x).
        if self.dropout:
            rate = self.dropout_rate
            #rate = np.clip(np.random.normal(self.dropout_rate, 1), 0.1, 0.9)
            z_mean = tf.layers.dropout(tf.layers.batch_normalization(z_mean), rate=rate)	
            z_log_var = tf.layers.dropout(tf.layers.batch_normalization(z_log_var), rate=rate)

        return(z_mean, z_log_var)

    def Convolutional_Encoding(self):
        """
        Encoder network when convolutional setting is set to True and convolutional 
        specifications are given.

        """
        weights = self.weights['encoder']
        biases = self.biases['encoder']

        inputs = tf.reshape(self.inputs, [-1, 10, 10, 10, 3])

        params = zip(sorted(weights.keys()), sorted(biases.keys()))
        encoding = inputs
        for p in params[:-2]:
            encoding = Convolution(encoding, weights[p[0]], biases[p[1]], maxpool=True)
        
        final_dim = self.convolutions[-1][-1]

        # (10x10x10) input => (5x5x5) ; round up (6x6x6x8) => (3x3x3x16) 
        encoding = tf.reshape(encoding, [-1, 3*3*3*final_dim])
        z_log_var = tf.add(tf.matmul(encoding, weights[params[-2][0]]), biases[params[-2][1]])
        z_mean = tf.add(tf.matmul(encoding, weights[params[-1][0]]), biases[params[-1][1]])
        return(z_mean, z_log_var)

    def Decode(self):
        """
        Takes compressed data from the latent space and reconstructs
        the data by passing it through a decoder network. The reparameterization
        trick is used, wherein Gaussian noise, N(0,1), is added to the latent
        representation so that backpropagation can flow through the parameters
        that govern the variational distribution. 

        Recall that in VAEs, the output is represented as a distribution over the
        feature space. The weights here aren't Bernoulli distributed, as is the case
        for the MNIST dataset, but range from [-1.0, 1.0]. For the purposes of creating a
        consistent loss function, we first convert to a Bernoulli distribution, then
        during simulation we move back into the range [-1.0, 1.0]
       
        """

        weights = self.weights['decoder']
        biases = self.biases['decoder']

        # ====================== TODO: USE RECURSION(); NO LOOPS  ======================

        # weight and bias indices have the same names
        params = zip(sorted(weights.keys()), sorted(biases.keys()))
        if self.new_graph:
            with open("decoder_keys", "a+") as f:
                print(params)
        if self.num_classes != 0:
            decoding = tf.concat(axis=1, values=[self.z, self.labels])
        else:
            decoding = self.z
        for p in params[:-1]:
            decoding = tf.add(tf.matmul(decoding, weights[p[0]]), biases[p[1]])
            if self.dropout:
                #rate = np.clip(np.random.normal(self.dropout_rate, 1), 0.1, 0.9)
                rate = self.dropout_rate
                decoding = self.activation(decoding)
                decoding = tf.layers.dropout(inputs=decoding, rate=rate)
            else:
                decoding = self.activation(decoding)

        output_mean = tf.nn.sigmoid(tf.add(tf.matmul(decoding, weights[params[-1][0]]), biases[params[-1][1]]))

        return(output_mean)

    def Optimize(self):
        """ 
        The output of the decoder should both resemble the input and achieve comparable performance
        on the given task.

        """
        # Two terms in the loss function: E[log(P(x | z))] - KL[Q(z | x)||P(z)] 
        # (i) reconstruction loss = E[log(P(x | z))]
        # (ii) latent loss = KL[Q(z | x)||P(z)]
        # Has closed form solution (see notes) given that both terms
        # are (multivariate) Gaussians.

        # Weight the loss function by the relative fitness of the newly constructed phenotype.
        # Because fitness values range from [0,1] this weight will serve the function of ensuring
        # that the hypothesized reconstruction has a similar degree of fitness when placed in the
        # appropriate environment.

        # L2 Regularization of weights:
        with tf.name_scope('l2_regularization'):
            regularized_loss = 0
            lambda_ = self.regularizer
            for key in ('encoder', 'decoder'):
                parameters = [tf.nn.l2_loss(param) for param in self.weights[key].values()]
                regularized_loss += lambda_*tf.add_n(parameters)
 
        with tf.name_scope('cost'):
            #latent_loss = (self.KL_Divergence() + self.first_KL)/2
            self.cost = tf.reduce_mean(self.Reconstruction_Loss() + self.KL_Divergence())
            self.cost += regularized_loss

        # GOAL: minimize the negative log likelihood as much as possible. Output of loss should
        # be INCREASINGLY negative as epochs go by.
        #self.optimizer = \
        #    tf.train.AdamOptimizer(learning_rate=VAE.HYPERPARAMS['learning_rate']).minimize(self.cost)

        with tf.name_scope("Adam_optimizer"):

            if not self.new_graph:
                # Adam optimizer is just silly. Load saved parameters from collection:
                beta1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam')[0]
                beta2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam')[1]
                method = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate,
                                                       beta1=beta1, beta2=beta2, name='newADAM')
            else:
                # Gradient clipping to prevent vanishing/exploding gradients:
                method = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
                #method = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = method.compute_gradients(self.cost)
                clipped_gradients = [(tf.clip_by_value(delta, -2.0, 2.0), param) for delta,param in gradients]
                self.optimizer = method.apply_gradients(clipped_gradients)



    def Reconstruction_Loss(self, offset=1e-7):
        """
        The first term in the evidence lower bound, which we seek to maximize.

        """
        # Canonical Bernoulli: (h^(x))*((1-h)^(1-x))
        # Log loss: x*(log(h)) + (1-x)(log(1-h))
        # Intuitively, this negative log probability can be envisaged as the nats required to reconstruct the
        # input. Expectation with respect to z~Q(z|x) is implicit if we sample enough data. "Monte Carlo" approx.
        # E[log(p(X | z))] where E[.] is wrt z~Q(z|x) and p(X | z) is just the output of the decoder
      
        # When we apply a sigmoid function to the output, we are tacitly assuming that the distribution is Bernoulli.
        # Axis = 1 sums across each instance e.g. tf.reduce_sum([[1,1,1], [1,1,1]]) = [3,3]

       # if self.dropout:
       #     output = tf.clip_by_value(self.output_mean, offset, 1-offset)
       #     output2 = tf.clip_by_value(self.output2_mean, offset, 1-offset)
       #     output_loss = -tf.reduce_sum(self.original*(tf.log(output)) + (1-self.original)*(tf.log(1 - output)), axis=1)
       #     output2_loss = -tf.reduce_sum(self.original*(tf.log(output2)) + (1-self.original)*(tf.log(1 - output2)), axis=1)
       #     return((output_loss + output2_loss)/2)
        #else: 
        output = tf.clip_by_value(self.output_mean, offset, 1-offset)
        return(-tf.reduce_sum(self.x*(tf.log(output)) + (1-self.x)*(tf.log(1 - output)), axis=1))

    def L1_Loss(self):
        return(tf.reduce_sum(tf.abs(self.output_mean - self.x)))

    def L2_Loss(self):
        # As an alternative to using sigmoidal output, MSE is appropriate for the assumption of
        # Gaussian distribution across features.
        return(tf.reduce_sum(tf.square(self.output_mean - self.x), axis=1))

    def KL_Divergence(self):
        """
        Occam's Razor: discourages overly complex distributions, Q(z | x).

        Ensures that our learned latent distribution closely resembles our prior, N(0,I). This
        constrains the representation such that the input data is clustered in space, and not
        assigned to a unique region.

        """
        return(-0.5*(tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1)))


    def Fit(self, X, labels=0):
        """
        Trains the model        

        """
        if isinstance(labels, int):
            opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        else:
            opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.labels: labels, self.x: X})
            
        return(cost)
    
    def Map(self, X, labels=0):
        """
        Map the input data to the latent space by running self.sess.run(self.z_mean...)
        This evaluates the tensor self.z_mean for the TF graph created using the functions above

        """
        self.dropout = False

        if isinstance(labels, int):
            return(self.sess.run(self.z_mean, feed_dict={self.x: X}))
        else:
            return(self.sess.run(self.z_mean, feed_dict={self.x: X, self.labels: labels}))

    def Generate(self, pop_size, labels=0, z_mean=0):
       
        # As per the VAE tutorial, we sample from the standard normal distribution N(0,I)
        # to generate new samples:

        self.dropout = False

        if isinstance(labels, int):
            if isinstance(z_mean, int):
                z_mean = np.random.normal(size=(pop_size, self.blueprint[-1]), loc=0, scale=1.0)
            return(self.sess.run(self.output_mean, feed_dict={self.z: z_mean}))

        else:
            if isinstance(z_mean, int):
                z_mean = np.random.normal(size=(pop_size, self.blueprint[-1]), loc=0, scale=1.0)
            return(self.sess.run(self.output_mean, feed_dict={self.z: z_mean, self.labels: labels}))

    def Reconstruct(self, X, labels=0):
        """
        Full pass through the network, from input to hypothesized/reconstructed
        output.
        
        """
        self.batch_size = batch_size = X.shape[0]
        self.dropout = False
        if isinstance(labels, int):
            return(self.sess.run(self.output_mean, feed_dict={self.x: X}))
        else:
            return(self.sess.run(self.output_mean, feed_dict={self.x: X, self.labels: labels}))

    def Save(self, name):

        timestamp = datetime.now().strftime(r"%y%m%d_%H%M")
        filename = os.path.join("{0}_VAE_{1}".format(timestamp, name))
        self.saver.save(self.sess, filename)
