import random
import numpy
from solution import solution

# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
class DE(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.mutation_factor = 0.5
        self.crossover_ratio = 0.9

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "DE"
        self.objfname = objf.__name__
        self.objf = objf
        self.sol_shift = sol_shift

        # convert lb, ub to array
        self.lb = numpy.array([lb for _ in range(dim)])
        self.ub = numpy.array([ub for _ in range(dim)])

        self.best = float("inf")

        # initialize population
        self.solutions = []

        for p in range(PopSize):
            sol = []
            for d in range(dim):
                d_val = random.uniform(self.lb[d], self.ub[d])
                sol.append(d_val)
            self.solutions.append(sol)
        self.solutions = numpy.array(self.solutions)
        self.population_fitness = []
        # calculate fitness for all the population
        for i in range(PopSize):
            fitness = objf(self.solutions[i, :]-self.sol_shift)
            self.population_fitness += [fitness]
            if fitness < self.best:
                self.best = fitness
                self.bestIndividual = self.solutions[i, :]
    
    def update(self, iter_id):
        if iter_id < self.maxiers:   
            for i in range(self.popnum):
                # 1. Mutation

                # select 3 random solution except current solution
                ids_except_current = [_ for _ in range(self.popnum) if _ != i]
                id_1, id_2, id_3 = random.sample(ids_except_current, 3)

                mutant_sol = []
                # for d in range(self.dim):
                d_val = self.solutions[id_1] + self.mutation_factor * (
                    self.solutions[id_2] - self.solutions[id_3]
                )

                # 2. Recombination
                rn = numpy.random.random(self.dim) > self.crossover_ratio#uniform(0, 1)
                # if rn > self.crossover_ratio:
                d_val[rn] = self.solutions[i,rn]

                # add dimension value to the mutant solution
                mutant_sol = d_val.copy()

                # 3. Replacement / Evaluation

                # clip new solution (mutant)
                mutant_sol = numpy.clip(mutant_sol, self.lb, self.ub)

                # calc fitness
                mutant_fitness = self.objf(mutant_sol-self.sol_shift)

                # replace if mutant_fitness is better
                if mutant_fitness < self.population_fitness[i]:
                    self.solutions[i, :] = mutant_sol
                    self.population_fitness[i] = mutant_fitness

                    # update leader
                    if mutant_fitness < self.best:
                        self.best = mutant_fitness
                        self.bestIndividual = mutant_sol

