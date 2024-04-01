import numpy as np
import random as rng

class HPChromosome:
    '''
    num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, num_dense, num_neurons, padding, activation_fun, pool_type, dropout, dropout_rate, batch_norm, learning_rate, epochs, batch_size, momentum, l1_norm_rate, optimizer, l2_pen
    '''

    # TODO: initialize with random values within ranges
    def __init__(self, num_conv = None, num_kernels = None, kernel_size = None, conv_stride = None, num_pooling = None, pool_size = None,
                 pool_stride = None, num_dense = None, num_neurons = None, padding = None, activation_fun = None, pool_type = None,
                 dropout = None, dropout_rate = None, batch_norm = None, learning_rate = None, epochs = None, batch_size = None,
                 momentum = None, l1_norm_rate = None, optimizer = None, l2_pen = None):

        self.num_conv = rng.randint(1,10) if num_conv is None else num_conv
        self.num_kernels = rng.randint(1,20) if num_kernels is None else num_kernels
        self.kernel_size = rng.randrange(3,7,2) if kernel_size is None else kernel_size
        self.conv_stride = rng.randint(0,5) if conv_stride is None else conv_stride
        self.num_pooling = rng.randint(0,5) if num_pooling is None else num_pooling
        self.pool_size = rng.randrange(2,8,2) if pool_size is None else pool_size
        self.pool_stride = rnd.randint(0,5) if pool_stride is None else pool_stride
        self.num_dense = rnd.randint(0,20) if num_dense is None else num_dense
        self.num_neurons = rnd.randint(100,1000) if num_neurons is None else num_neurons
        self.padding = rnd.randint(0,5) if padding is None else padding
        self.activation_fun = rnd.randint(0,3) if activation_fun is None else activation_fun
        self.pool_type = rnd.randint(0,2) if pool_type is None else pool_type
        self.dropout = rnd.randint(0,1) if dropout is None else dropout
        self.dropout_rate = rnd.random() if dropout_rate is None else dropout_rate
        self.batch_norm = rnd.randint(0,1) if batch_norm is None else batch_norm
        self.learning_rate = rnd.uniform(0.00001, 1) if learning_rate is None else learning_rate
        self.epochs = rnd.randint(1,50) if epochs is None else epochs
        self.batch_size = rnd.randint((20,100) if batch_size is None else batch_size
        self.momentum = rnd.random() if momentum is None else momentum
        self.l1_norm_rate = rnd.uniform(0.00001, 1) if l1_norm_rate is None else l1_norm_rate
        self.optimizer = rnd.randint(0,2) if optimizer is None else optimizer
        self.l2_pen = rnd.uniform(0.00001, 1) if l2_pen is None else l2_pen
