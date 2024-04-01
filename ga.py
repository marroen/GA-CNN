from hp import HPChromosome
import numpy as np

def init(n):
    print("init")
    population = create_population(n)
    print("pop. created")
    print(population)

def create_population(n):
    population = np.empty(0)
    for _ in range(n):
        population = np.append(population, HPChromosome())
    return population

def crossover():
    print("crossover")

def selection():
    print("selection")

def tournament():
    print("tournament)"

