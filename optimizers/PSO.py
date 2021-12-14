import random
import numpy
from solution import solution

class PSO(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.Vmax = 6
        self.wMax = 0.9
        self.wMin = 0.2
        self.c1 = 2
        self.c2 = 2

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "PSO"
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

        ######################## Initializations

        self.vel = numpy.zeros((PopSize, dim))
        self.pBest = numpy.zeros((PopSize, dim))
        self.pBestScore = numpy.zeros((PopSize))+float("inf")

    def update(self, iter_id):
        if iter_id < self.maxiers:   
            w = self.wMax - iter_id * ((self.wMax - self.wMin) / self.maxiers)
            for i in range(self.popnum):
                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)
                # Calculate objective function for each particle
                fitness = self.objf(self.solutions[i, :]-self.sol_shift)

                if self.pBestScore[i] > fitness:
                    self.pBestScore[i] = fitness
                    self.pBest[i, :] = self.solutions[i, :].copy()

                if self.best > fitness:
                    self.best = fitness
                    self.bestIndividual = self.solutions[i, :].copy()

            # Update solutions
            for i in range(self.popnum):
                r1 = numpy.random.random(self.dim)
                r2 = numpy.random.random(self.dim)
                # for j in range(self.dim):
                self.vel[i] = (
                    w * self.vel[i]
                    + self.c1 * r1 * (self.pBest[i] - self.solutions[i])
                    + self.c2 * r2 * (self.bestIndividual - self.solutions[i])
                )

                self.vel[i] = numpy.clip(self.vel[i], -self.Vmax, self.Vmax)

                self.solutions[i] += self.vel[i]
