import random
import numpy
import math
from solution import solution


class SSA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "SSA"
        self.objfname = objf.__name__
        self.objf = objf
        self.sol_shift = sol_shift

        # convert lb, ub to array
        self.lb = numpy.array([lb for _ in range(dim)])
        self.ub = numpy.array([ub for _ in range(dim)])

        self.best = float("inf")

        # initialize population
        self.solutions = []
        self.solutions_new = numpy.zeros((self.popnum,self.dim))

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
        self.population_fitness = numpy.array(self.population_fitness)

    def update(self, iter_id):
        if iter_id < self.maxiers:   

            # Number of flames Eq. (3.14) in the paper
            # Flame_no=round(N-Iteration*((N-1)/Max_iteration));

            c1 = 2 * math.exp(-((4 * iter_id / self.maxiers) ** 2))
            # Eq. (3.2) in the paper

            for i in range(self.popnum):
                if i < self.popnum // 2:
                    for j in range(self.dim):
                        c2 = random.random()
                        c3 = random.random()
                        # Eq. (3.1) in the paper
                        if c3 < 0.5:
                            self.solutions[i, j] = self.bestIndividual[j] + c1 * (
                                (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                            )
                        else:
                            self.solutions[i, j] = self.bestIndividual[j] - c1 * (
                                (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                            )

                        ####################

                elif i >= self.popnum // 2 and i < self.popnum + 1:
                    point1 = self.solutions[i - 1]
                    point2 = self.solutions[i]

                    self.solutions[i] = (point2 + point1) / 2
                    # Eq. (3.4) in the paper

            for i in range(self.popnum):

                # Check if salps go out of the search spaceand bring it back
                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)

                self.population_fitness[i] = self.objf(self.solutions[i, :]-self.sol_shift)

                if self.population_fitness[i] < self.best:
                    self.bestIndividual = numpy.copy(self.solutions[i, :])
                    self.best = self.population_fitness[i]

