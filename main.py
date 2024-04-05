import cnn
import ga
from hp import HPChromosome

def main():
    ga.init(20, 20, 0.20)
    '''
    hps = []
    for _ in range(20):
        #BASELINE
        hp = HPChromosome(num_conv = 5, num_kernels = 16, kernel_size = 3, conv_stride = 1, num_pooling = 2, pool_size = 2, pool_stride = 2,
                          num_dense= 4, num_neurons= 120, padding=1, activation_fun=0, pool_type= 0, dropout = 1, dropout_rate = 0.2,
                          batch_norm =1, learning_rate = 0.001, epochs = 10, batch_size = 64, momentum = 0.9, l1_norm_rate=0.001,
                          optimizer = 0, l2_pen=1e-5)
        
        #BEST MODEL
        hp = HPChromosome(num_conv = 2, num_kernels = 5, kernel_size = 3, conv_stride = 1, num_pooling = 1, pool_size = 2, pool_stride = 1,
                          num_dense= 1, num_neurons= 140, padding=0, activation_fun=2, pool_type= 0, dropout = 0, dropout_rate = 0.3686321124596793,
                          batch_norm = 1, learning_rate = 0.013207272115559677, epochs = 41, batch_size = 47, momentum = 0.5227651886353928, l1_norm_rate=0.07556563213918935,
                          optimizer = 0, l2_pen=0.08302143717655085)
        
        hp_fit = cnn.cnn_parameterized(hp)
        hps.append(hp_fit)
        print("Output:", hp_fits
    print("Average:", sum(hps)/20)
    '''
    

if __name__ == "__main__":
    main()
