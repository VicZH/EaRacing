import numpy
import random
from solution import solution

class BAT(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        # Loudness  (constant or decreasing)
        self.A = 0.5
        # Pulse rate (constant or decreasing)
        self.r = 0.5
        
        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "BAT"
        self.objfname = objf.__name__
        self.objf = objf
        self.sol_shift = sol_shift

        # convert lb, ub to array
        self.lb = numpy.array([lb for _ in range(dim)])
        self.ub = numpy.array([ub for _ in range(dim)])
        self.best = float("inf")

        self.Qmin = 0  # Frequency minimum
        self.Qmax = 2  # Frequency maximum

        # Initializing arrays
        self.Q = numpy.zeros(self.popnum)  # Frequency
        self.v = numpy.zeros((self.popnum, self.dim))  # Velocities
        self.S = numpy.zeros((self.popnum, self.dim))  # new solutions

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
            # Loop over all bats(solutions)
            for i in range(0, self.popnum):
                self.Q[i] = self.Qmin + (self.Qmin - self.Qmax) * random.random()
                self.v[i, :] = self.v[i, :] + (self.solutions[i, :] - self.bestIndividual) * self.Q[i]
                self.S[i, :] = self.solutions[i, :] + self.v[i, :]

                # Check boundaries
                for j in range(self.dim):
                    self.solutions[i, j] = numpy.clip(self.solutions[i, j], self.lb[j], self.ub[j])

                # Pulse rate
                if random.random() > self.r:
                    self.S[i, :] = self.bestIndividual + 0.001 * numpy.random.randn(self.dim)

                # Evaluate new solutions
                Fnew = self.objf(self.S[i, :]-self.sol_shift)

                # Update if the solution improves
                if (Fnew <= self.population_fitness[i]) and (random.random() < self.A):
                    self.solutions[i, :] = numpy.copy(self.S[i, :])
                    self.population_fitness[i] = Fnew

                # Update the current best solution
                if Fnew <= self.best:
                    self.bestIndividual = numpy.copy(self.S[i, :])
                    self.best = Fnew
