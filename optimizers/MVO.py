
import random
import numpy
import math
from numpy import asarray
from sklearn.preprocessing import normalize
from solution import solution


def normr(Mat):
    """normalize the columns of the matrix
    B= normr(A) normalizes the row
    the dtype of A is float"""
    Mat = Mat.reshape(1, -1)
    # Enforce dtype float
    if Mat.dtype != "float":
        Mat = asarray(Mat, dtype=float)

    # if statement to enforce dtype float
    B = normalize(Mat, norm="l2", axis=1)
    B = numpy.reshape(B, -1)
    return B


def randk(t):
    if (t % 2) == 0:
        s = 0.25
    else:
        s = 0.75
    return s


def RouletteWheelSelection(weights):
    accumulation = numpy.cumsum(weights)
    p = random.random() * accumulation[-1]
    chosen_index = -1
    for index in range(0, len(accumulation)):
        if accumulation[index] > p:
            chosen_index = index
            break

    choice = chosen_index

    return choice


class MVO(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.WEP_Max = 1
        self.WEP_Min = 0.2

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "MVO"
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
        self.Sorted_universes = numpy.zeros((self.popnum,self.dim))

    def update(self, iter_id):
        if iter_id < self.maxiers:

            "Eq. (3.3) in the paper"
            WEP = self.WEP_Min + iter_id * ((self.WEP_Max - self.WEP_Min) / self.maxiers)

            TDR = 1 - (math.pow(iter_id, 1 / 6) / math.pow(self.maxiers, 1 / 6))

            for i in range(self.popnum):
                self.solutions[i] = numpy.clip(self.solutions[i], self.lb, self.ub)

                self.population_fitness[i] = self.objf(self.solutions[i, :]-self.sol_shift)

                if self.population_fitness[i] < self.best:

                    self.best = self.population_fitness[i]
                    self.bestIndividual = numpy.array(self.solutions[i, :])

            sorted_Inflation_rates = numpy.sort(self.population_fitness)
            sorted_indexes = numpy.argsort(self.population_fitness)

            self.Sorted_universes = self.solutions[sorted_indexes]

            normalized_sorted_Inflation_rates = numpy.copy(normr(sorted_Inflation_rates))

            self.solutions[0, :] = self.Sorted_universes[0, :].copy()

            for i in range(self.popnum):
                Back_hole_index = i
                for j in range(self.dim):
                    r1 = random.random()

                    if r1 < normalized_sorted_Inflation_rates[i]:
                        White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates)

                        if White_hole_index == -1:
                            White_hole_index = 0
                        White_hole_index = 0
                        self.solutions[Back_hole_index, j] = self.Sorted_universes[
                            White_hole_index, j
                        ]

                    r2 = random.random()

                    if r2 < WEP:
                        r3 = random.random()
                        if r3 < 0.5:
                            self.solutions[i, j] = self.bestIndividual[j] + TDR * (
                                (self.ub[j] - self.lb[j]) * random.random() + self.lb[j]
                            )  # random.uniform(0,1)+lb);
                        if r3 > 0.5:
                            self.solutions[i, j] = self.bestIndividual[j] - TDR * (
                                (self.ub[j] - self.lb[j]) * random.random() + self.lb[j]
                            )  # random.uniform(0,1)+lb);

