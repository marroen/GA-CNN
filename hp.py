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

        self.num_conv = rng.randint(1,5) if num_conv is None else 5
        self.num_kernels = rng.randint(1,10) if num_kernels is None else 16
        self.kernel_size = rng.randrange(3,7,2) if kernel_size is None else 3
        self.conv_stride = rng.randint(1,3) if conv_stride is None else 1
        self.num_pooling = rng.randint(0,2) if num_pooling is None else 2
        self.pool_size = rng.randrange(2,8,2) if pool_size is None else 2
        self.pool_stride = rng.randint(1,2) if pool_stride is None else 2
        self.num_dense = rng.randint(1,10) if num_dense is None else 4
        self.num_neurons = rng.randint(100,500) if num_neurons is None else 120
        self.padding = rng.randint(0,5) if padding is None else 1
        self.activation_fun = rng.randint(0,3) if activation_fun is None else 0
        self.pool_type = rng.randint(0,1) if pool_type is None else 0
        self.dropout = rng.randint(0,1) if dropout is None else 1
        self.dropout_rate = rng.random() if dropout_rate is None else 0.2
        self.batch_norm = rng.randint(0,1) if batch_norm is None else 1
        self.learning_rate = rng.uniform(0.00001, 1) if learning_rate is None else 0.001
        self.epochs = rng.randint(1,50) if epochs is None else 10
        self.batch_size = rng.randint(20,100) if batch_size is None else 64
        self.momentum = rng.random() if momentum is None else 0.9
        self.l1_norm_rate = rng.uniform(0.00001, 1) if l1_norm_rate is None else 0.001
        self.optimizer = rng.randint(0,2) if optimizer is None else 0
        self.l2_pen = rng.uniform(0.00001, 1) if l2_pen is None else 0.00001

