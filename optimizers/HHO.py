"""
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________

"""
import random
import numpy
import math
from solution import solution

def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step

class HHO(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        # initialize the location and Energy of the rabbit
        self.Rabbit_Location = numpy.zeros(dim)
        self.Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "HHO"
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
            for i in range(self.popnum):
                self.solutions[i, :] = numpy.clip(self.solutions[i, :], self.lb, self.ub)

                # fitness of locations
                fitness = self.objf(self.solutions[i, :]-self.sol_shift)

                # Update the location of Rabbit
                if fitness < self.Rabbit_Energy:  # Change this to > for maximization problem
                    self.Rabbit_Energy = fitness
                    self.Rabbit_Location = self.solutions[i, :].copy()

                if fitness < self.best:
                    self.best = fitness
                    self.bestIndividual = self.solutions[i, :].copy()

            E1 = 2 * (1 - (iter_id / self.maxiers))  # factor to show the decreaing energy of rabbit

            # Update the location of Harris' hawks
            for i in range(self.popnum):

                E0 = 2 * random.random() - 1  # -1<E0<1
                Escaping_Energy = E1 * (
                    E0
                )  # escaping energy of rabbit Eq. (3) in the paper

                # -------- Exploration phase Eq. (1) in paper -------------------

                if abs(Escaping_Energy) >= 1:
                    # Harris' hawks perch randomly based on 2 strategy:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.popnum * random.random())
                    X_rand = self.solutions[rand_Hawk_index, :]
                    if q < 0.5:
                        # perch based on other family members
                        self.solutions[i, :] = X_rand - random.random() * abs(
                            X_rand - 2 * random.random() * self.solutions[i, :]
                        )

                    elif q >= 0.5:
                        # perch on a random tall tree (random site inside group's home range)
                        self.solutions[i, :] = (self.Rabbit_Location - self.solutions.mean(0)) - random.random() * (
                            (self.ub - self.lb) * random.random() + self.lb
                        )

                # -------- Exploitation phase -------------------
                elif abs(Escaping_Energy) < 1:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r = random.random()  # probablity of each event

                    if (
                        r >= 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (6) in paper
                        self.solutions[i, :] = (self.Rabbit_Location) - Escaping_Energy * abs(
                            self.Rabbit_Location - self.solutions[i, :]
                        )

                    if (
                        r >= 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (4) in paper
                        Jump_strength = 2 * (
                            1 - random.random()
                        )  # random jump strength of the rabbit
                        self.solutions[i, :] = (self.Rabbit_Location - self.solutions[i, :]) - Escaping_Energy * abs(
                            Jump_strength * self.Rabbit_Location - self.solutions[i, :]
                        )

                    # phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if (
                        r < 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (10) in paper
                        # rabbit try to escape by many zigzag deceptive motions
                        Jump_strength = 2 * (1 - random.random())
                        X1 = self.Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * self.Rabbit_Location - self.solutions[i, :]
                        )
                        X1 = numpy.clip(X1, self.lb, self.ub)

                        if self.objf(X1-self.sol_shift) < fitness:  # improved move?
                            self.solutions[i, :] = X1.copy()
                        else:  # hawks perform levy-based short rapid dives around the rabbit
                            X2 = (
                                self.Rabbit_Location
                                - Escaping_Energy
                                * abs(Jump_strength * self.Rabbit_Location - self.solutions[i, :])
                                + numpy.multiply(numpy.random.randn(self.dim), Levy(self.dim))
                            )
                            X2 = numpy.clip(X2, self.lb, self.ub)
                            if self.objf(X2-self.sol_shift) < fitness:
                                self.solutions[i, :] = X2.copy()
                    if (
                        r < 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (11) in paper
                        Jump_strength = 2 * (1 - random.random())
                        X1 = self.Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * self.Rabbit_Location - self.solutions.mean(0)
                        )
                        X1 = numpy.clip(X1, self.lb, self.ub)

                        if self.objf(X1-self.sol_shift) < fitness:  # improved move?
                            self.solutions[i, :] = X1.copy()
                        else:  # Perform levy-based short rapid dives around the rabbit
                            X2 = (
                                self.Rabbit_Location
                                - Escaping_Energy
                                * abs(Jump_strength * self.Rabbit_Location - self.solutions.mean(0))
                                + numpy.multiply(numpy.random.randn(self.dim), Levy(self.dim))
                            )
                            X2 = numpy.clip(X2, self.lb, self.ub)
                            if self.objf(X2-self.sol_shift) < fitness:
                                self.solutions[i, :] = X2.copy()

