""" Sine Cosine OPtimization Algorithm """

import random
import numpy
import math
from solution import solution

class SCA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "SCA"
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
            for i in range(self.popnum):

                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)

                # Calculate objective function for each search agent
                fitness = self.objf(self.solutions[i] - self.sol_shift)

                if fitness < self.best:
                    self.best = fitness  # Update Dest_Score
                    self.bestIndividual = self.solutions[i].copy()

            # Eq. (3.4)
            a = 2
            r1 = a - iter_id * ((a) / self.maxiers)  # r1 decreases linearly from a to 0

            # Update the Position of search agents
            for i in range(self.popnum):
                for j in range(self.dim):

                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = (2 * numpy.pi) * random.random()
                    r3 = 2 * random.random()
                    r4 = random.random()

                    # Eq. (3.3)
                    if r4 < (0.5):
                        # Eq. (3.1)
                        self.solutions[i, j] = self.solutions[i, j] + (
                            r1 * numpy.sin(r2) * abs(r3 * self.bestIndividual[j] - self.solutions[i, j])
                        )
                    else:
                        # Eq. (3.2)
                        self.solutions[i, j] = self.solutions[i, j] + (
                            r1 * numpy.cos(r2) * abs(r3 * self.bestIndividual[j] - self.solutions[i, j])
                        )

