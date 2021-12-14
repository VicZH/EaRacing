""" JAYA Algorithm """

import random
import numpy
from solution import solution

class JAYA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        # Worst position initialization

        self.Worst_pos = numpy.zeros(dim)
        self.Worst_score = float(0)

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "JAYA"
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
                self.bestIndividual = self.solutions[i].copy()
            if fitness > self.Worst_score:
                self.Worst_score = fitness
                self.Worst_pos = self.solutions[i].copy()
        self.population_fitness = numpy.array(self.population_fitness)
    
    def update(self, iter_id):
        if iter_id < self.maxiers:   

            # Update the Position of search agents
            for i in range(self.popnum):
                New_Position = numpy.zeros(self.dim)
                
                r1 = numpy.random.random(self.dim)
                r2 = numpy.random.random(self.dim)

                # JAYA Equation
                New_Position = (
                    self.solutions[i]
                    + r1 * (self.bestIndividual - abs(self.solutions[i]))
                    - r2 * (self.Worst_pos - abs(self.solutions[i]))
                )

                New_Position = numpy.clip(New_Position,self.lb,self.ub)

                new_fitness = self.objf(New_Position-self.sol_shift)

                # replacing current element with new element if it has better fitness
                if new_fitness < self.population_fitness[i]:
                    self.solutions[i] = New_Position.copy()
                    self.population_fitness[i] = new_fitness

            # finding the best and worst element
            for i in range(self.popnum):
                if self.population_fitness[i] < self.best:
                    self.best = self.population_fitness[i]
                    self.bestIndividual = self.solutions[i].copy()

                if self.population_fitness[i] > self.Worst_score:
                    self.Worst_score = self.population_fitness[i]
                    self.Worst_pos = self.solutions[i].copy()
