#% ======================================================== %
#% Firefly Algorithm for constrained optimization:          %
#% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
#% Second Edition, Luniver Press, (2010).   www.luniver.com %
#% ======================================================== %

import numpy
import math, random
from solution import solution


def alpha_new(alpha, NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta = 1 - (10 ** (-4) / 0.9) ** (1 / NGen)
    alpha = (1 - delta) * alpha
    return alpha


class FFA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.alpha = 0.5  # Randomness 0--1 (highly random)
        self.betamin = 0.20  # minimum value of beta
        self.gamma = 1  # Absorption coefficient

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "FFA"
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

            #% This line of reducing alpha is optional
            self.alpha = alpha_new(self.alpha, self.maxiers)

            #% Evaluate new solutions (for all n fireflies)
            zn = numpy.zeros(self.popnum)
            Lightn = numpy.zeros(self.popnum)
            for i in range(self.popnum):
                zn[i] = self.objf(self.solutions[i, :]-self.sol_shift)
                Lightn[i] = zn[i]

            # Ranking fireflies by their light intensity/objectives
            Lightn = numpy.sort(zn)
            Index = numpy.argsort(zn)
            ns = self.solutions[Index, :]

            # Find the current best
            nso = ns
            Lighto = Lightn
            nbest = ns[0, :]
            Lightbest = Lightn[0]

            #% For output only
            fbest = Lightbest

            #% Move all fireflies to the better locations
            #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
            #          Lightbest,alpha,betamin,gamma,Lb,Ub);
            scale = []
            for b in range(self.dim):
                scale.append(abs(self.ub[b] - self.lb[b]))
            scale = numpy.array(scale)
            for i in range(self.popnum):
                # The attractiveness parameter beta=exp(-gamma*r)
                for j in range(self.popnum):
                    r = numpy.sqrt(numpy.sum((ns[i, :] - ns[j, :]) ** 2))
                    # r=1
                    # Update moves
                    if Lightn[i] > Lighto[j]:  # Brighter and more attractive
                        beta0 = 1
                        beta = (beta0 - self.betamin) * math.exp(-self.gamma * r ** 2) + self.betamin
                        tmpf = self.alpha * (numpy.random.rand(self.dim) - 0.5) * scale
                        ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf

            # calculate fitness for all the population
            for i in range(self.popnum):
                fitness = self.objf(ns[i, :]-self.sol_shift)
                if fitness < self.population_fitness[i]:
                    self.population_fitness[i] = fitness
                    self.solutions[i] = ns[i].copy()
                    if fitness < self.best:
                        self.best = fitness
                        self.bestIndividual = self.solutions[i]