import sys

from ga.fitness import Fitness
from ga.fitness_graph import plot_fitness_graph
import ga.initial_pop as pop
import ga.selection as selection
import ga.reproduce as reproduce
import ga.output as output
from ga.custom_types import Population
from utils import *


#TODO: define a gait length, for training / getting fitness and one for output
# since training is optimising the sine waves so it does not actually need to be the full 300 speeding up training time

# define the set of frames in the gait cycle
gait_length:int = 50

#NOTE: as the above todo says, we could use a var like this I will implement tonight
#gait_length_output:int = 300

# define search space
population_size:int = 3000
mutation_rate:float = 0.025
crossover_rate:float = 0.7
output_file:str = "results.txt"
# define fitness target score
fitness_score_target:float = 1.0

defaults = {
    "gait_length":gait_length,
    "population_size": population_size,
    "mutation_rate": mutation_rate,
    "crossover_rate": crossover_rate,
    "output_file": output_file,
    "fitness_score_target": fitness_score_target
}

def main(gait_length:int = gait_length,population_size:int = population_size,mutation_rate:float = mutation_rate, crossover_rate:float = crossover_rate, output_file:str = output_file,fitness_score_target:float = fitness_score_target):
    '''
    Main file to run genetic algorithm for gait generation of the spider.
    The Genetic algorithm evolves the population over a defined set number of generations and outputs the best solution found.
    Draws a fitness graph at the end showing best and average fitness scores over generations.
    A max population size and gait length can be defined to control the search space.
    '''

    population: Population = pop.gen_population(population_size)
    # create fitness object using the class from fitness module(python file)
    fit: Fitness = Fitness(gait_length)
    # lists to store the best fitness scores over generations for plotting
    fitness_over_time: list[float] = []
    # list to store average fitness scores over generations for plotting
    avg_fitness_over_time: list[float] = []
    # run GA for set number of generations defined above
    gen:int = 0
    current_best_fitness:float = 0.0
    while current_best_fitness < fitness_score_target:
        # stores the fitness score of each individual in the population
        # stores the index of the best individual in current population
        fitness_list: list[float] = []  
        # generate fitness list
        best_idx: int = 0
        # go through each individual in the population and get the fitness score 
        for individual in population:
            # calculate fitness score of the current individual using the method from fitness class
            individual_fitness: float = fit.get_fitness(individual)
            # append fitness score to list
            fitness_list.append(individual_fitness)
        # get index of best individual in current population
        best_idx: int = fitness_list.index(max(fitness_list))
        # update current best fitness
        current_best_fitness: float = fitness_list[best_idx]
        # print information about current generation
        gen += 1
        print("generation:",gen,"| best index: ",best_idx, "| fitness: ",fitness_list[best_idx])

        # calculate average fitness for current generation
        avg_fitness: float = sum(fitness_list) / len(fitness_list)
        # append best and average fitness to their respective lists
        fitness_over_time.append(fitness_list[best_idx])
        avg_fitness_over_time.append(avg_fitness)

        if len(fitness_over_time) >= 100:
            last_100_avg: float = sum(fitness_over_time[-100:]) / 100
            if round(last_100_avg , 3) == round(current_best_fitness, 3):
                print("Fitness target consistently met over 100 generations with the best fitness score being:", current_best_fitness)
                sys.exit(0)

        # select individuals for next generation using the functions from selection module(python file)
        population = selection.tournament(population,fitness_list,10)
        # perform crossover and mutation to generate new individuals using functions from reproduce module(python file)
        population = reproduce.uniform_crossover(population,gait_length,crossover_rate)
        population = reproduce.mutate(population,mutation_rate)


    # output the best individual found after the genetic algorithm is terminated
    output.output_gait(output_file,population[best_idx],300)

    # plot fitness graph using the function from fitness_graph module(python file)
    plot_fitness_graph(fitness_over_time, avg_fitness_over_time, gen)


if __name__ == "__main__":
    main()
