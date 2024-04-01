from hp import HPChromosome
import numpy as np
import random as rng

def init(n):
    print("init")
    population = create_population(n)
    print("pop. created")

def create_population(n):
    population = np.empty(0)
    for _ in range(n):
        population = np.append(population, HPChromosome())
    return population

# TODO: replace with heuristic for ranges, uniform
def crossover(parent1, parent2):
    total_hp = 22
    child1 = HPChromosome()
    child2 = HPChromosome()
    print("crossover")
    for i in range(total_hp):
        rnd = rng.random()
        chosen_parent = parent1 if round(rnd) == 1 else parent2
        other_parent = parent2 if round(rnd) == 1 else parent1
        match i:
            case 0:
                child1.num_conv = chosen_parent.num_conv
                child2.num_conv = other_parent.num_conv
            case 1:
                child1.num_kernels = chosen_parent.num_kernels
                child2.num_kernels = other_parent.num_kernels
            case 2:
                child1.kernel_size = chosen_parent.kernel_size
                child2.kernel_size = other_parent.kernel_size
            case 3:
                child1.conv_stride = chosen_parent.conv_stride
                child2.conv_stride = other_parent.conv_stride
            case 4:
                child1.num_pooling = chosen_parent.num_pooling
                child2.num_pooling = other_parent.num_pooling
            case 5:
                child1.pool_size = chosen_parent.pool_size
                child2.pool_size = other_parent.pool_size
            case 6:
                child1.pool_stride = chosen_parent.pool_stride
                child2.pool_stride = other_parent.pool_stride
            case 7:
                child1.num_dense = chosen_parent.num_dense
                child2.num_dense = other_parent.num_dense
            case 8:
                child1.num_neurons = chosen_parent.num_neurons
                child2.num_neurons = other_parent.num_neurons
            case 9:
                child1.padding = chosen_parent.padding
                child2.padding = other_parent.padding
            case 10:
                child1.activation_fun = chosen_parent.activation_fun
                child2.activation_fun = other_parent.activation_fun
            case 11:
                child1.pool_type = chosen_parent.pool_type
                child2.pool_type = other_parent.pool_type
            case 12:
                child1.dropout = chosen_parent.dropout
                child2.dropout = other_parent.dropout
            case 13:
                child1.dropout_rate = chosen_parent.dropout_rate
                child2.dropout_rate = other_parent.dropout_rate
            case 14:
                child1.batch_norm = chosen_parent.batch_norm
                child2.batch_norm = other_parent.batch_norm
            case 15:
                child1.learning_rate = chosen_parent.learning_rate
                child2.learning_rate = other_parent.learning_rate
            case 16:
                child1.epochs = chosen_parent.epochs
                child2.epochs = other_parent.epochs
            case 17:
                child1.batch_size = chosen_parent.batch_size
                child2.batch_size = other_parent.batch_size
            case 18:
                child1.momentum = chosen_parent.momentum
                child2.momentum = other_parent.momentum
            case 19:
                child1.l1_norm_rate = chosen_parent.l1_norm_rate
                child2.l1_norm_rate = other_parent.l1_norm_rate
            case 20:
                child1.optimizer = chosen_parent.optimizer
                child2.optimizer = other_parent.optimizer
            case 21:
                child1.l2_pen = chosen_parent.l2_pen
                child2.l2_pen = other_parent.l2_pen

def selection():
    print("selection")

def tournament():
    print("tournament")

