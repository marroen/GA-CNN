import cnn
import ga
from hp import HPChromosome

def main():
    ga.init(10)
    '''
    hp = HPChromosome(num_conv = 5, num_kernels = 16, kernel_size = 3, conv_stride = 1, num_pooling = 2, pool_size = 2, pool_stride = 2,
                      num_dense= 4, num_neurons= 120, padding=1, activation_fun=0, pool_type= 0, dropout = 1, dropout_rate = 0.2,
                      batch_norm =1, learning_rate = 0.001, epochs = 10, batch_size = 64, momentum = 0.9, l1_norm_rate=0.001,
                      optimizer = 0, l2_pen=1e-5)
    print("Output:", cnn.cnn_parameterized(hp))
    '''

if __name__ == "__main__":
    main()
