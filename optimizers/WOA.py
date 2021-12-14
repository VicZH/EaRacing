
import random
import numpy
import math
from solution import solution

class WOA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):
        
        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "WOA"
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

                # Return back the search agents that go beyond the boundaries of the search space
                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)

                # Calculate objective function for each search agent
                fitness = self.objf(self.solutions[i, :]-self.sol_shift)

                # Update the leader
                if fitness < self.best:  # Change this to > for maximization problem
                    self.best = fitness
                    # Update alpha
                    self.bestIndividual = self.solutions[i,:].copy()  # copy current whale position into the leader position

            a = 2 - iter_id * ((2) / self.maxiers)
            # a decreases linearly fron 2 to 0 in Eq. (2.3)

            # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + iter_id * ((-1) / self.maxiers)

            # Update the Position of search agents
            for i in range(self.popnum):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                C = 2 * r2  # Eq. (2.4) in the paper

                b = 1
                #  parameters in Eq. (2.5)
                l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)

                p = random.random()  # p in Eq. (2.6)

                for j in range(self.dim):

                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = math.floor(
                                self.popnum * random.random()
                            )
                            X_rand = self.solutions[rand_leader_index, :]
                            D_X_rand = abs(C * X_rand[j] - self.solutions[i, j])
                            self.solutions[i, j] = X_rand[j] - A * D_X_rand

                        elif abs(A) < 1:
                            D_Leader = abs(C * self.bestIndividual[j] - self.solutions[i, j])
                            self.solutions[i, j] = self.bestIndividual[j] - A * D_Leader

                    elif p >= 0.5:

                        distance2Leader = abs(self.bestIndividual[j] - self.solutions[i, j])
                        # Eq. (2.5)
                        self.solutions[i, j] = (
                            distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                            + self.bestIndividual[j]
                        )
