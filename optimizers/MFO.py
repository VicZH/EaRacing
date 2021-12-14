
import random
import numpy
import math
from solution import solution

class MFO(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "MFO"
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

        self.sorted_population = numpy.copy(self.solutions)
        self.fitness_sorted = numpy.zeros(self.popnum)
        #####################
        self.best_flames = numpy.copy(self.solutions)
        self.best_flame_fitness = numpy.zeros(self.popnum)
        ####################
        self.double_population = numpy.zeros((2 * self.popnum, dim))
        self.double_fitness = numpy.zeros(2 * self.popnum)

        self.double_sorted_population = numpy.zeros((2 * self.popnum, dim))
        self.double_fitness_sorted = numpy.zeros(2 * self.popnum)
        #########################
        self.previous_population = numpy.zeros((self.popnum, dim))
        self.previous_fitness = numpy.zeros(self.popnum)

    def update(self, iter_id):
        if iter_id < self.maxiers:

            # Number of flames Eq. (3.14) in the paper
            Flame_no = round(self.popnum - iter_id * ((self.popnum - 1) / self.maxiers))

            for i in range(self.popnum):

                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)

                # evaluate moths
                self.population_fitness[i] = self.objf(self.solutions[i, :]-self.sol_shift)

            if iter_id == 1:
                # Sort the first population of moths
                self.fitness_sorted = numpy.sort(self.population_fitness).copy()
                I = numpy.argsort(self.population_fitness)

                self.sorted_population = self.solutions[I, :].copy()

                # Update the flames
                self.best_flames = self.sorted_population.copy()
                self.best_flame_fitness = self.fitness_sorted.copy()
            else:
                #
                #        # Sort the moths
                self.double_population = numpy.concatenate(
                    (self.previous_population.copy(), self.best_flames.copy()), axis=0
                )
                self.double_fitness = numpy.concatenate(
                    (self.previous_fitness.copy(), self.best_flame_fitness.copy()), axis=0
                )
                #
                self.double_fitness_sorted = numpy.sort(self.double_fitness).copy()
                I2 = numpy.argsort(self.double_fitness).copy()
                #
                #
                for newindex in range(0, 2 * self.popnum):
                    self.double_sorted_population[newindex, :] = numpy.array(
                        self.double_population[I2[newindex], :]
                    )

                self.fitness_sorted = self.double_fitness_sorted[0:self.popnum]
                self.sorted_population = self.double_sorted_population[0:self.popnum, :]
                #
                #        # Update the flames
                self.best_flames = self.sorted_population.copy()
                self.best_flame_fitness = self.fitness_sorted.copy()

            #
            #   # Update the position best flame obtained so far
            self.best = self.fitness_sorted[0]
            self.bestIndividual = self.sorted_population[0, :]
            #
            self.previous_population = self.solutions
            self.previous_fitness = self.population_fitness
            #
            # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + iter_id * ((-1) / self.maxiers)

            # Loop counter
            for i in range(self.popnum):
                #
                for j in range(self.dim):
                    if (
                        i <= Flame_no
                    ):  # Update the position of the moth with respect to its corresponsing flame
                        #
                        # D in Eq. (3.13)
                        distance_to_flame = abs(self.sorted_population[i, j] - self.solutions[i, j])
                        b = 1
                        t = (a - 1) * random.random() + 1
                        #
                        #                % Eq. (3.12)
                        self.solutions[i, j] = (
                            distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi)
                            + self.sorted_population[i, j]
                        )
                    #            end
                    #
                    if (
                        i > Flame_no
                    ):  # Upaate the position of the moth with respct to one flame
                        #
                        #                % Eq. (3.13)
                        distance_to_flame = abs(self.sorted_population[i, j] - self.solutions[i, j])
                        b = 1
                        t = (a - 1) * random.random() + 1
                        #
                        #                % Eq. (3.12)
                        self.solutions[i, j] = (
                            distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi)
                            + self.sorted_population[Flame_no, j]
                        )

