# Kareem Mohamed Morsy Ismail, ID: 20190386, Group: CS-S3, Program: CS
# David Emad Philip Ata-Allah, ID: 20190191, Group: CS-S3, Program: CS

import random

ITERATIONS = 50000
POPULATION_SIZE = 500
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
ELITISM_RATE = 0.01
TOURNAMENT_SIZE = 5
UPPER_BOUND = 10
LOWER_BOUND = -10
DEPENDENCY_FACTOR = 3


def initializePopulation(populationSize, d):
    population = []
    for _ in range(populationSize):
        population.append([random.uniform(LOWER_BOUND, UPPER_BOUND)
                           for _ in range(d + 1)])
    return population


def calculateMSE(individual, data):
    def calculateIndividual(individual, x):
        return sum([theta * x ** i for i, theta in enumerate(individual)])

    error = 0
    for x, y in data:
        error += (y - calculateIndividual(individual, x)) ** 2
    return error / len(data)


def calculateFitness(individual, data):
    return 1 / calculateMSE(individual, data)


def selectParent(population, fitness):
    # Tournament selection
    tournament = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournamentFitness = [fitness[i] for i in tournament]
    i = tournament[tournamentFitness.index(max(tournamentFitness))]
    return i, population[i]


def crossover(parent1, parent2):
    # Crossover only if random number is less than crossover rate
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    # 2-point crossover
    r1 = random.randint(0, len(parent1) - 1)
    r2 = random.randint(0, len(parent2) - 1)
    if r1 > r2:
        r1, r2 = r2, r1

    child1 = parent1[:r1] + parent2[r1:r2] + parent1[r2:]
    child2 = parent2[:r1] + parent1[r1:r2] + parent2[r2:]

    return child1, child2


def mutate(individual, t):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Non-uniform mutation
            delta_lower = individual[i] - LOWER_BOUND
            delta_upper = UPPER_BOUND - individual[i]

            deciding_factor = random.random()

            if deciding_factor <= 0.5:
                y = delta_lower
            else:
                y = delta_upper

            delta = y * (1 - random.random() ** (
                (1 - t / ITERATIONS) ** DEPENDENCY_FACTOR))

            if deciding_factor <= 0.5:
                individual[i] -= delta
            else:
                individual[i] += delta

    return individual


def getEliteList(fitness):
    return sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)


def isElite(fitness, i, eliteList):
    elite = eliteList[:int(ELITISM_RATE * len(fitness))]
    return i in elite


def randomNonElite(fitness, eliteList):
    nonElite = eliteList[int(ELITISM_RATE * len(fitness)):]
    return random.choice(nonElite)


def doWork(d, data):
    population = initializePopulation(POPULATION_SIZE, d)
    
    # Evaluate fitness
    fitness = [calculateFitness(individual, data)
                for individual in population]

    for t in range(ITERATIONS):
        # Number of families
        families = random.randint(0, POPULATION_SIZE // 2)
        for _ in range(families):
            # Selection
            i1, parent1 = selectParent(population, fitness)
            i2, parent2 = selectParent(population, fitness)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutate
            child1 = mutate(child1, t)
            child2 = mutate(child2, t)

            # Replacement
            eliteList = getEliteList(fitness)

            inplace1 = not isElite(fitness, i1, eliteList)
            inplace2 = not isElite(fitness, i2, eliteList)
            if not inplace1:
                i1 = randomNonElite(fitness, eliteList)
                
            population[i1] = child1
            fitness[i1] = calculateFitness(child1, data)

            if not inplace2:
                i2 = randomNonElite(fitness, eliteList)
                
            population[i2] = child2
            fitness[i2] = calculateFitness(child2, data)

        print("Iteration: {}, MSE: {}".format(t, 1 / fitness[eliteList[0]]))

    best_individual = population[fitness.index(max(fitness))]
    return best_individual, calculateMSE(best_individual, data)


with open("curve_fitting_input.txt") as f:
    data = f.readlines()

    # Reading data line by line
    t = int(data[0])
    current_line = 1
    for i in range(t):
        line = data[current_line].split()
        n = int(line[0])
        d = int(line[1])
        current_line += 1

        data_points = []

        for j in range(n):
            line = data[current_line].split()
            x = float(line[0])
            y = float(line[1])
            current_line += 1
            data_points.append((x, y))

        coefficients, mse = doWork(d, data_points)

        # Append the coefficients to the output file
        with open("curve_fitting_output.txt", "a") as f:
            f.write("Dataset {}:\n".format(i + 1))
            f.write("Coefficients: ")
            f.write(", ".join([str(c) for c in coefficients]))
            f.write("\n")
            f.write("MSE: {}\n\n".format(mse))
