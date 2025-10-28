from fitness import *
import ga.initial_pop as pop
import selection
import reproduce
import output

'''
Main file to run genetic algorithm for gait generation of the spider.
The Genetic algorithm evolves the population over a defined set number of generations and outputs the best solution found.
A max population size and gait length can be defined to control the search space.
'''
def main():
    # define number of generations to run GA for
    generations:int =  1350
    # define the set of frames in the gait cycle
    gait_length = 40
    # define search space
    max_pop = 1300
    # generate initial population using a function from the initial_pop module(python file)
    population = pop.gen_population(max_pop,gait_length)
    # create fitness object using the class from fitness module(python file)
    fit = fitness(gait_length)
    # run GA for set number of generations defined above
    for gen in range(generations):
        # stores the fitness score of each individual in the population
        # generate fitness list
        fitness_list = []
        # stores the index of the best individual in current population
        best_idx = 0
        # go through each individual in the population and get the fitness score 
        for individual in population:
            # calculate fitness score of the current individual using the method from fitness class
            individual_fitness = fit.get_fitness(individual)
            # append fitness score to list
            fitness_list.append(individual_fitness)
        # get index of best individual in current population
        best_idx = fitness_list.index(max(fitness_list))
        # print information about current generation
        print("generation:",gen,"| best index: ",best_idx, "| fitness: ",fitness_list[best_idx])

        # select individuals for next generation using the functions from selection module(python file)
        population = selection.tournament(population,fitness_list,3)
        # perform crossover and mutation to generate new individuals using functions from reproduce module(python file)
        population = reproduce.uniform_crossover(population,gait_length,0.7)
        population = reproduce.mutate(population,0.02)

    # output the best individual found after the genetic algorithm is terminated
    output.output("results.txt",population[best_idx])


if __name__ == "__main__":
    main()
