import numpy
import random
import math
from solution import solution

class CS(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        # Discovery rate of alien eggs/solutions
        self.pa = 0.25

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "CS"
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

    def update(self, iter_id):
        if iter_id < self.maxiers: 
            # Generate new solutions (but keep the current best)
            # get_cuckoos - perform Levy flights
            beta = 3 / 2
            sigma = (
                math.gamma(1 + beta)
                * math.sin(math.pi * beta / 2)
                / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)

            s = numpy.zeros(self.dim)
            for j in range(0, self.popnum):
                s = self.solutions[j, :]
                u = numpy.random.randn(len(s)) * sigma
                v = numpy.random.randn(len(s))
                step = u / abs(v) ** (1 / beta)

                stepsize = 0.01 * (step * (s - self.bestIndividual))

                s = s + stepsize * numpy.random.randn(len(s))

                for k in range(self.dim):
                    self.solutions_new[j,k] = numpy.clip(s[k], self.lb[k], self.ub[k])

            # Replace some solutions by constructing new solutions
            # Discovered or not
            tempnest = numpy.zeros((self.popnum, self.dim))

            K = numpy.random.uniform(0, 1, (self.popnum, self.dim)) > self.pa

            stepsize = random.random() * (
                self.solutions[numpy.random.permutation(self.popnum), :] - self.solutions[numpy.random.permutation(self.popnum), :]
            )

            tempnest = self.solutions + stepsize * K

            # Evaluate new solutions and find best
            for i in range(self.popnum):
                fitness = self.objf(tempnest[i, :]-self.sol_shift)
                if fitness < self.population_fitness[i]:
                    self.population_fitness[i] = fitness
                    self.solutions[i] = tempnest[i].copy()
                    if fitness < self.best:
                        self.best = fitness
                        self.bestIndividual = self.solutions[i]
