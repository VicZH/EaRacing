import random
import numpy
from solution import solution

class GWO(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):
        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "GWO"
        self.objfname = objf.__name__
        self.objf = objf
        self.sol_shift = sol_shift

        # initialize alpha, beta, and delta_pos
        self.Alpha_pos = numpy.zeros(dim)
        self.Alpha_score = float("inf")

        self.Beta_pos = numpy.zeros(dim)
        self.Beta_score = float("inf")

        self.Delta_pos = numpy.zeros(dim)
        self.Delta_score = float("inf")

        # convert lb, ub to array
        self.lb = numpy.array([lb for _ in range(dim)])
        self.ub = numpy.array([ub for _ in range(dim)])
        self.best = float("inf")

        # Initialize the positions of search agents
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
            if fitness < self.best:
                self.best = fitness
                self.bestIndividual = self.solutions[i, :]
 
    def update(self, iter_id):

        for i in range(self.popnum):
            # Calculate objective function for each search agent
            fitness = self.objf(self.solutions[i, :]-self.sol_shift)

            # Update Alpha, Beta, and Delta
            if fitness < self.Alpha_score:
                self.Delta_score = self.Beta_score  # Update delte
                self.Delta_pos = self.Beta_pos.copy()
                self.Beta_score = self.Alpha_score  # Update beta
                self.Beta_pos = self.Alpha_pos.copy()
                self.Alpha_score = fitness
                # Update alpha
                self.Alpha_pos = self.solutions[i, :].copy()

            if fitness > self.Alpha_score and fitness < self.Beta_score:
                self.Delta_score = self.Beta_score  # Update delte
                self.Delta_pos = self.Beta_pos.copy()
                self.Beta_score = fitness  # Update beta
                self.Beta_pos = self.solutions[i, :].copy()

            if fitness > self.Alpha_score and fitness > self.Beta_score and fitness < self.Delta_score:
                self.Delta_score = fitness  # Update delta
                self.Delta_pos = self.solutions[i, :].copy()

        a = 2 - iter_id * ((2) / self.maxiers)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(self.popnum):
            r1 = numpy.random.random(self.dim)  # r1 is a random number in [0,1]
            r2 = numpy.random.random(self.dim)  # r2 is a random number in [0,1]
            # for j in range(self.dim):
            A1 = 2 * a * r1 - a
            # Equation (3.3)
            C1 = 2 * r2
            # Equation (3.4)

            D_alpha = numpy.abs(C1 * self.Alpha_pos - self.solutions[i])
            # Equation (3.5)-part 1
            X1 = self.Alpha_pos - A1 * D_alpha
            # Equation (3.6)-part 1

            r1 = numpy.random.random(self.dim)  # r1 is a random number in [0,1]
            r2 = numpy.random.random(self.dim)  # r2 is a random number in [0,1]

            A2 = 2 * a * r1 - a
            # Equation (3.3)
            C2 = 2 * r2
            # Equation (3.4)

            D_beta = numpy.abs(C2 * self.Beta_pos - self.solutions[i])
            # Equation (3.5)-part 2
            X2 = self.Beta_pos - A2 * D_beta
            # Equation (3.6)-part 2

            r1 = numpy.random.random(self.dim)  # r1 is a random number in [0,1]
            r2 = numpy.random.random(self.dim)  # r2 is a random number in [0,1]

            A3 = 2 * a * r1 - a
            # Equation (3.3)
            C3 = 2 * r2
            # Equation (3.4)

            D_delta = numpy.abs(C3 * self.Delta_pos - self.solutions[i])
            # Equation (3.5)-part 3
            X3 = self.Delta_pos - A3 * D_delta
            # Equation (3.5)-part 3

            self.solutions[i] = (X1 + X2 + X3) / 3  # Equation (3.7)

        # self.best = self.Alpha_score
        # self.bestIndividual = self.Alpha_pos.copy()

        for i in range(self.popnum):
            self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)
            fitness = self.objf(self.solutions[i, :]-self.sol_shift)
            if fitness < self.best:
                self.best = fitness
                self.bestIndividual = self.solutions[i, :]
