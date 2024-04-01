import numpy as np

class HPChromosome:
    '''
    num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, num_dense, num_neurons, padding, activation_fun, pool_type, dropout, dropout_rate, batch_norm, learning_rate, epochs, batch_size, momentum, l1_norm_rate, optimizer, l2_pen
    '''

    # TODO: initialize with random values within ranges
    def __init__(self, num_conv = None, num_kernels = None, kernel_size = None, conv_stride = None, num_pooling = None, pool_size = None,
                 pool_stride = None, num_dense = None, num_neurons = None, padding = None, activation_fun = None, pool_type = None,
                 dropout = None, dropout_rate = None, batch_norm = None, learning_rate = None, epochs = None, batch_size = None,
                 momentum = None, l1_norm_rate = None, optimizer = None, l2_pen = None):

        self.num_conv = 0 if num_conv is None else num_conv
        self.num_kernels = 0 if num_kernels is None else num_kernels
        self.kernel_size = 0 if kernel_size is None else kernel_size
        self.conv_stride = 0 if conv_stride is None else conv_stride
        self.num_pooling = 0 if num_pooling is None else num_pooling
        self.pool_size = 0 if pool_size is None else pool_size
        self.pool_stride = 0 if pool_stride is None else pool_stride
        self.num_dense = 0 if num_dense is None else num_dense
        self.num_neurons = 0 if num_neurons is None else num_neurons
        self.padding = 0 if padding is None else padding
        self.activation_fun = 0 if activation_fun is None else activation_fun
        self.pool_type= 0 if pool_type is None else pool_type
        self.dropout = 0 if dropout is None else dropout
        self.dropout_rate = 0 if dropout_rate is None else dropout_rate
        self.batch_norm = 0 if batch_norm is None else batch_norm
        self.learning_rate = 0 if learning_rate is None else learning_rate
        self.epochs = 0 if epochs is None else epochs
        self.batch_size = 0 if batch_size is None else batch_size
        self.momentum = 0 if momentum is None else momentum
        self.l1_norm_rate = 0 if l1_norm_rate is None else l1_norm_rate
        self.optimizer = 0 if optimizer is None else optimizer
        self.l2_pen = 0 if l2_pen is None else l2_pen
            
