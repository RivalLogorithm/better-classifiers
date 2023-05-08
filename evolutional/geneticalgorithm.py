import random
from random import randint, uniform


class Chromosome:
    def __init__(self, function, genes):
        self.genes = genes
        self.function = function

        self.fitness = self.function(self.genes)


class GeneticAlgorithm:
    def __init__(self, function, population_size, generations, varbound, vartype):
        self.best_solution = None
        self.function = function
        self.population_size = population_size
        self.generations = generations
        self.varbound = varbound
        self.vartype = vartype
        self.initial_population = self.generate_initial_population()

    def run(self):
        for _ in range(self.generations):
            i1 = randint(0, self.population_size - 1)
            i2 = randint(0, self.population_size - 1)
            while i1 == i2:
                i2 = randint(0, self.population_size - 1)

            child_population = self.crossover(self.initial_population[i1], self.initial_population[i2])
            self.initial_population += child_population
            self.initial_population = sorted(self.initial_population, key=lambda x: x.fitness[1], reverse=True)
            self.initial_population = self.initial_population[:len(self.initial_population) - 2]

            self.best_solution = self.initial_population[0]
        return self.best_solution.fitness

    def generate_initial_population(self):
        population = []
        while len(population) < self.population_size:
            arguments =[]
            for i in range(len(self.varbound)):
                if self.vartype[i] == 'int':
                    arguments.append(randint(self.varbound[i][0], self.varbound[i][1]))
                elif self.vartype[i] == 'real':
                    arguments.append(uniform(self.varbound[i][0], self.varbound[i][1]))
                elif self.vartype[i] == 'str':
                    arguments.append(random.choice(self.varbound[i]))
                    # arguments.append(self.varbound[i][randint(0, len(self.varbound[i]))])
                elif self.vartype[i] == 'bool':
                    arguments.append(random.choice(self.varbound[i]))
            population.append(Chromosome(self.function, arguments))
        return population

    def crossover(self, parent1, parent2):
        br_point = randint(0, len(parent1.genes) - 1)
        child1 = self.generate_child(parent1.genes, parent2.genes, br_point)
        child2 = self.generate_child(parent2.genes, parent1.genes, br_point)

        chromosome1 = Chromosome(self.function, child1)
        chromosome2 = Chromosome(self.function, child2)
        return [chromosome1, chromosome2]


    def generate_child(self, parent1, parent2, br_point):
        child = []
        for i in range(0, br_point):
            child.append(parent1[i])
        for i in range(br_point, len(parent1)):
            child.append(parent2[i])
        return child
