
import numpy
import random
from solution import solution


def elitism(population, scores, bestIndividual, bestScore):
    """
    This melitism operator of the population
    """
    # get the worst individual
    worstFitnessId = selectWorstIndividual(scores)
    # replace worst cromosome with best one from previous generation if its fitness is less than the other
    if scores[worstFitnessId] > bestScore:
        population[worstFitnessId] = numpy.copy(bestIndividual)
        scores[worstFitnessId] = numpy.copy(bestScore)

def selectWorstIndividual(scores):
    """
    It is used to get the worst individual in a population based n the fitness value
    """
    maxFitnessId = numpy.where(scores == numpy.max(scores))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId



class GA(solution):
    def __init__(self, objf, sol_shift, lb, ub, dim, PopSize, EvlNum):

        self.cp = 1  # crossover Probability
        self.mp = 0.01  # Mutation Probability
        self.keep = 2
        # elitism parameter: how many of the best individuals to keep from one generation to the next

        self.dim = dim
        self.popnum = PopSize
        self.maxiers = int(EvlNum / PopSize)

        self.optimizer = "GA"
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

            # The crossover of all individuals
            # initialize a new population
            newPopulation = numpy.empty_like(self.solutions)
            newPopulation[0:self.keep] = self.solutions[0:self.keep]
            # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
            for i in range(self.keep, self.popnum, 2):
                # pair of parents selection
                ##reverse score because minimum value should have more chance of selection
                reverse = max(self.population_fitness) + min(self.population_fitness)
                reverseScores = reverse - self.population_fitness.copy()
                sumScores = sum(reverseScores)
                pick = random.uniform(0, sumScores)
                current = 0
                for individualId in range(self.popnum):
                    current += reverseScores[individualId]
                    if current > pick:
                        parent1Id = individualId
                        break
                pick = random.uniform(0, sumScores)
                current = 0
                for individualId in range(self.popnum):
                    current += reverseScores[individualId]
                    if current > pick:
                        parent2Id = individualId
                        break
                parent1 = self.solutions[parent1Id].copy()
                parent2 = self.solutions[parent2Id].copy()
                crossoverLength = min(len(parent1), len(parent2))
                parentsCrossoverProbability = random.uniform(0.0, 1.0)
                if parentsCrossoverProbability < self.cp:
                    # The point at which crossover takes place between two parents.
                    crossover_point = random.randint(0, crossoverLength - 1)
                    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
                    offspring1 = numpy.concatenate(
                        [parent1[0:crossover_point], parent2[crossover_point:]]
                    )
                    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
                    offspring2 = numpy.concatenate(
                        [parent2[0:crossover_point], parent1[crossover_point:]]
                    )
                else:
                    offspring1 = parent1.copy()
                    offspring2 = parent2.copy()

                # Add offsprings to population
                newPopulation[i] = numpy.copy(offspring1)
                newPopulation[i + 1] = numpy.copy(offspring2)

            # mutation
            for i in range(self.keep, self.popnum):
                # Mutation
                offspringMutationProbability = random.uniform(0.0, 1.0)
                if offspringMutationProbability < self.mp:
                    mutationIndex = random.randint(0, self.dim - 1)
                    mutationValue = random.uniform(self.lb[mutationIndex], self.ub[mutationIndex])
                    newPopulation[i, mutationIndex] = mutationValue


            # ga = clearDups(ga, lb, ub)
            newPopulation_unique = numpy.unique(newPopulation, axis=0)
            oldLen = len(newPopulation)
            newLen = len(newPopulation_unique)
            if newLen < oldLen:
                nDuplicates = oldLen - newLen
                newPopulation_unique = numpy.append(
                    newPopulation_unique,
                    numpy.random.uniform(0, 1, (nDuplicates, self.dim))
                    * (numpy.array(self.ub) - numpy.array(self.lb))
                    + numpy.array(self.lb),
                    axis=0,
                )

            # Loop through individuals in population
            for i in range(self.popnum):
                # Return back the search agents that go beyond the boundaries of the search space
                newPopulation_unique[i] = numpy.clip(newPopulation_unique[i], self.lb, self.ub)

            # calculate fitness for all the population
            for i in range(self.popnum):
                fitness = self.objf(newPopulation_unique[i, :]-self.sol_shift)
                if fitness < self.population_fitness[i]:
                    self.population_fitness[i] = fitness
                    self.solutions[i] = newPopulation_unique[i].copy()
                    if fitness < self.best:
                        self.best = fitness
                        self.bestIndividual = self.solutions[i]

            # Sort from best to worst
            sortedIndices = self.population_fitness.argsort()
            self.solutions = self.solutions[sortedIndices]
            self.population_fitness = self.population_fitness[sortedIndices]

