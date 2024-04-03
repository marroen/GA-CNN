from hp import HPChromosome
from cnn import cnn_parameterized
import numpy as np
import random as rng

def init(n):
    print("init")
    pop_fit_map = create_pop_fit_map(create_pop(n)) 
    print("pop. created")
    run(pop_fit_map)

def run(pop_fit_map):
    for i in range(5):
        print("Generation", i)
        #pop_fit_map = selection(pop_fit_map)
        selection(pop_fit_map)
        print("Average fitness: ", get_average_fit(pop_fit_map))
    print("Final best HP:", max(pop_fit_map.values()))

def create_pop(n):
    pop = np.empty(0)
    for _ in range(n):
        pop = np.append(pop, HPChromosome())
    return pop

def create_pop_fit_map(pop):
    hp_fits = {}
    for hp in pop:
        print("num_conv:", hp.num_conv)
        print("num_kernels:", hp.num_kernels)
        print("kernel_size:", hp.kernel_size)
        print("conv_stride:", hp.conv_stride)
        print("num_pooling:", hp.num_pooling)
        print("pool_size:", hp.pool_size)
        print("pool_stride:", hp.pool_stride)
        print("num_dense:", hp.num_dense)
        print("num_neurons:", hp.num_neurons)
        print("padding:", hp.padding)
        print("activation_fun:", hp.activation_fun)
        print("pool_type:", hp.pool_type)
        print("dropout:", hp.dropout)
        print("dropout_rate:", hp.dropout_rate)
        print("batch_norm:", hp.batch_norm)
        print("learning_rate:", hp.learning_rate)
        print("epochs:", hp.epochs)
        print("batch_size:", hp.batch_size)
        print("momentum:", hp.momentum)
        print("l1_norm_rate:", hp.l1_norm_rate)
        print("optimizer:", hp.optimizer)
        print("l2_pen:", hp.l2_pen)
        hp_fits[hp] = cnn_parameterized(hp)
    return hp_fits

def get_average_fit(pop_fit_map):
    return sum(pop_fit_map.values()) / len(pop_fit_map)

def selection(pop_fit_map):
    # After running, len(selected) = n
    #selected = []
    keys = list(pop_fit_map.keys())
    rng.shuffle(keys)
    print("selection")
    for i in range(len(keys)-1):
        if (i % 2 == 0):
            # Select two parents from shuffled keys list
            mother = (keys[i], pop_fit_map[keys[i]])
            father = (keys[i+1], pop_fit_map[keys[i+1]])
            # Get children based on just parent keys
            children = crossover(keys[i], keys[i+1])
            # Match parents, with known fitness, against children
            winners = tournament(mother, father, children)
            # Update winners to population
            pop_fit_map.pop(keys[i])
            pop_fit_map.pop(keys[i+1])
            for winner in winners:
                pop_fit_map[winner[0]] = winner[1]
    print("Ensuring len(pop_fit_map) == n:", len(pop_fit_map))
    #return selected

def tournament(mother, father, children):
    child1_fitness = cnn_parameterized(children[0])
    child2_fitness = cnn_parameterized(children[1])
    print("tournament")
    
    first = (children[0], child1_fitness)
    second = (children[1], child2_fitness)
    parents = [mother, father]
    for parent in parents:
        if parent[1] > first[1]:
            second = first
            first = parent
        elif parent[1] > second[1]:
            second = parent
    return [first, second]

def mutate():
    print("mutate")

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
    return [child1, child2]

    '''
    num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, num_dense, num_neurons, padding, activation_fun, pool_type, dropout, dropout_rate, batch_norm, learning_rate, epochs, batch_size, momentum, l1_norm_rate, optimizer, l2_pen
    '''

