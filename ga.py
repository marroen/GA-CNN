from hp import HPChromosome
from cnn import cnn_parameterized
import numpy as np
import random as rng

def init(n, m, mutation_rate):
    print("----- CREATING POPULATION -----")
    pop_fit_map = create_pop_fit_map(create_pop(n)) 
    print("----- POPULATION CREATED -----")
    run(pop_fit_map, m, mutation_rate)

def run(pop_fit_map, m, mutation_rate):
    average_fits = []
    print("----- RUN STARTING -----")
    for i in range(m):
        print("----- GENERATION STARTING -----")
        print("Number", i)
        selection(pop_fit_map, mutation_rate)
        print("----- AVERAGE FITNESS ----- ")
        average_fit = get_average_fit(pop_fit_map)
        print(average_fit)
        average_fits.append(average_fit)
        print("----- GENERATION ENDING -----")

    print("----- RUN ENDED -----")

    best_key = list(reversed(sorted(pop_fit_map, key=lambda k: pop_fit_map[k])))[0]
    best_value = pop_fit_map[best_key]

    print("----- ANALYZING BEST HP -----")

    print("Final best HP:", best_value)
    print("Best HP values:")
    print_hp_values(best_key)

    print("----- ANALYZED BEST HP -----")

    print("----- ANALYZING TOP 5 HP -----")

    print("HP values and fitnesses of top 5:")
    i = 0
    for key in reversed(sorted(pop_fit_map, key=lambda k: pop_fit_map[k])):
        if i <= 5:
            value = pop_fit_map[key]
            print("----- ANALYZING TOP -----")
            print("Number", i)
            print("HP fitness:", value)
            print("HP values:")
            print_hp_values(key)
            print("----- ANALYZED TOP -----")
        i += 1

    print("----- ANALYZED TOP 5 HP -----")

    print("----- ANALYZING BOTTOM 5 HP -----")

    print("HP values and fitnesses of bottom 5:")
    i = 0
    for key in sorted(pop_fit_map, key=lambda k: pop_fit_map[k]):
        if i <= 5:
            value = pop_fit_map[key]
            print("----- ANALYZING BOTTOM -----")
            print("Number", i)
            print("HP fitness:", value)
            print("HP values:")
            print_hp_values(key)
            print("----- ANALYZED BOTTOM -----")
        i += 1
    print("----- ANALYZED BOTTOM 5 HP -----")

    print("----- AVERAGE FITNESSES -----")
    for average_fit in average_fits:
        print(average_fit)

def print_hp_values(hp):
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

def create_pop(n):
    pop = np.empty(0)
    for _ in range(n):
        pop = np.append(pop, HPChromosome())
    return pop

def create_pop_fit_map(pop):
    hp_fits = {}
    for hp in pop:
        #print_hp_values(hp)
        hp_fits[hp] = cnn_parameterized(hp)
    return hp_fits

def get_average_fit(pop_fit_map):
    return sum(pop_fit_map.values()) / len(pop_fit_map)

def selection(pop_fit_map, mutation_rate):
    keys = list(pop_fit_map.keys())
    rng.shuffle(keys)
    for i in range(len(keys)-1):
        if (i % 2 == 0):
            # Select two parents from shuffled keys list
            mother = (keys[i], pop_fit_map[keys[i]])
            father = (keys[i+1], pop_fit_map[keys[i+1]])
            # (Possibly) mutate parents
            mutate(keys[i], mutation_rate)
            mutate(keys[i+1], mutation_rate)
            # Get children based on just parent keys
            children = crossover(keys[i], keys[i+1])
            # Match parents, with known fitness, against children
            winners = tournament(mother, father, children)
            # Update winners to population
            pop_fit_map.pop(keys[i])
            pop_fit_map.pop(keys[i+1])
            for winner in winners:
                pop_fit_map[winner[0]] = winner[1]
    #print("Ensuring len(pop_fit_map) == n:", len(pop_fit_map))

def tournament(mother, father, children):
    child1_fitness = cnn_parameterized(children[0])
    child2_fitness = cnn_parameterized(children[1])
    
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

def mutate(hp, mutation_rate):
    total_hp = 22
    if rng.random() < mutation_rate:
        rnd = rng.random()
        i = rng.randint(0, 22)
        match i:
            case 0:
                hp.num_conv = rng.randint(1,5)
            case 1:
                hp.num_kernels = rng.randint(1,20)
            case 2:
                hp.kernel_size = rng.randrange(3,5,2)
            case 3:
                hp.conv_stride = rng.randint(1,2)
            case 4:
                hp.num_pooling = rng.randint(0,2)
            case 5:
                hp.pool_size = 2
            case 6:
                hp.pool_stride = rng.randint(1,2)
            case 7:
                hp.num_dense = rng.randint(1,5)
            case 8:
                hp.num_neurons = rng.randint(100, 200)
            case 9:
                hp.padding = rng.randint(0,5)
            case 10:
                hp.activation_fun = rng.randint(0,3)
            case 11:
                hp.pool_type = rng.randint(0,1)
            case 12:
                hp.dropout = rng.randint(0,1)
            case 13:
                hp.dropout_rate = rng.uniform(0.2, 0.5)
            case 14:
                hp.batch_norm = rng.randint(0,1)
            case 15:
                hp.learning_rate = rng.uniform(0.00001, 1)
            case 16:
                hp.epochs = rng.randint(1,50)
            case 17:
                hp.batch_size = rng.randint(20,100)
            case 18:
                hp.momentum = rng.uniform(0.5, 0.95)
            case 19:
                hp.l1_norm_rate = rng.uniform(0.00001, 1)
            case 20:
                hp.optimizer = rng.randint(0,2)
            case 21:
                hp.l2_pen = rng.uniform(0.00001, 1)
        
def crossover(parent1, parent2):
    total_hp = 22
    child1 = HPChromosome()
    child2 = HPChromosome()
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

