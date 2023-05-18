import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm


class Vector:
    def __init__(self, function, params):
        self.function = function
        self.params = params

        self.fitness = self.function(self.params)

    def __add__(self, other):
        result = copy.deepcopy(self)
        for i in range(len(self.params)):
            if type(self.params[i]) is not str and type(self.params[i]) is not bool:
                result.params[i] = self.params[i] + other.params[i]
            else:
                result.params[i] = self.params[i]
        return result

    def __sub__(self, other):
        result = copy.deepcopy(self)
        for i in range(len(self.params)):
            if type(self.params[i]) is not str and type(self.params[i]) is not bool:
                result.params[i] = self.params[i] - other.params[i]
            else:
                result.params[i] = self.params[i]
        return result

    def __rmul__(self, other):
        result = copy.deepcopy(self)
        for i in range(len(self.params)):
            if type(self.params[i]) is not str and type(self.params[i]) is not bool:
                result.params[i] = other * self.params[i]
            else:
                result.params[i] = self.params[i]
        return result


class DifferentialEvolution:
    def __init__(self, function, population_size, generations, crossover_rate, F, division_count, varbound, vartype):
        random.seed(123)
        self.function = function
        self.population_size = population_size
        self.varbound = varbound
        self.vartype = vartype
        self.initial_population = self.generate_initial_population()
        self.scaler = MinMaxScaler()
        self.division_count = division_count
        self.divisions = []
        self.F = F
        self.crossover_rate = crossover_rate
        for _ in range(division_count):
            self.divisions.append([])
        self.generations = generations

    def run(self):
        p = ProcessPoolExecutor(6)
        current_g = 0
        history = []
        for g in tqdm(range(self.generations)):
            current_best = self.divide()
            current_g = g + 1
            history.append(current_best)
            if current_best == -1:
                break
            self.initial_population = []
            results = []
            for i in range(self.division_count):
                result = p.submit(self.diff_evolve, self.divisions[i])
                results.append(result)
            for idx, res in enumerate(as_completed(results)):
                self.divisions[idx] = res.result()
                self.initial_population += self.divisions[idx]
        p.shutdown()
        self.initial_population = sorted(self.initial_population, key=lambda x: x.fitness[0])
        return self.initial_population[0].fitness, current_g, history,

    def diff_evolve(self, division):
        new_generation = []
        if len(division) < 4:
            return division
        for x in range(len(division)):
            random_list = [i for i in range(len(division)) if i != x]
            ids = random.sample(random_list, 3)
            m_v = self.do_mutation(division[ids[0]], division[ids[1]], division[ids[2]])
            try:
                tr_v = self.do_crossover(division[x], m_v)
            except Exception:
                new_generation.append(division[x])
                continue
            if tr_v.fitness[0] < division[x].fitness[0]:
                new_generation.append(tr_v)
            else:
                new_generation.append(division[x])
        return new_generation

    def do_mutation(self, v1, v2, v3):
        v = v1 + self.F * (v2 - v3)
        return v

    def do_crossover(self, x, v):
        r_ind = random.randint(0, len(x.params))
        for i in range(len(x.params)):
            r = random.uniform(0, 1)
            if r < self.crossover_rate or i == r_ind:
                v.params[i] = x.params[i]
        if self.function.__name__ == 'one_class_svm_func' and v.params[2] > 5:
            v.params[2] = 5
        return Vector(self.function, v.params)

    def generate_initial_population(self):
        population = []
        futures = []
        p = ProcessPoolExecutor(6)
        for _ in range(self.population_size):
            arguments = []
            for i in range(len(self.varbound)):
                if self.vartype[i] == 'int':
                    arguments.append(random.randint(self.varbound[i][0], self.varbound[i][1]))
                elif self.vartype[i] == 'real':
                    arguments.append(random.uniform(self.varbound[i][0], self.varbound[i][1]))
                elif self.vartype[i] == 'str':
                    arguments.append(random.choice(self.varbound[i]))
                elif self.vartype[i] == 'bool':
                    arguments.append(random.choice(self.varbound[i]))
            result = p.submit(init_vector, self.function, arguments)
            futures.append(result)
        for future in as_completed(futures):
            population.append(future.result())
        p.shutdown()
        return population

    def divide(self):
        fit = list(map(lambda x: x.fitness[0], self.initial_population))
        avg_fitness = np.mean(fit)
        std_dev = np.std(fit)
        f_min = np.min(fit)
        treshholds = []
        for i in range(1, self.division_count):
            treshholds.append(avg_fitness - (i - 1) * (std_dev - avg_fitness + f_min) / (self.division_count-2))
        treshholds.append(np.max(fit))
        treshholds.reverse()
        pop_level = []
        for _ in range(len(fit)):
            pop_level.append(0)
        for i in range(len(fit)):
            for j in range(self.division_count):
                if fit[i] <= treshholds[j]:
                    pop_level[i] = j

        divisions = []
        for _ in range(self.division_count):
            divisions.append([])

        for id, lvl in enumerate(pop_level):
            divisions[lvl].append(self.initial_population[id])

        self.divisions = divisions
        return sorted(self.initial_population, key=lambda x: x.fitness[0])[0].fitness[0]

def init_vector(func, params):
    return Vector(func, params)