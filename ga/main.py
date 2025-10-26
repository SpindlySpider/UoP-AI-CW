from inital_pop import Population_obj
from fitness import fitness
import selection
import reproduce
import output
import numpy

def main():
    generations:int =  300
    elites = 50
    population = Population_obj(10).gen_population()
    print(len(population[0][0]))
    fit = fitness()
    best_inx = 0
    for gen in range(generations):
        fitness_list = []
        # generate fitness list
        fit_val = 0
        for idx,individual in enumerate(population):
            fit_val:float = fit.leg_angles(individual)
            fitness_list.append(fit_val)
        # need to use numpy as python default max cannot handle multi-dimensional array
        best_idx = fitness_list.index(numpy.max(fitness_list))
        print(population[best_idx])
        print(gen,best_idx, fitness_list[best_idx])
        # select new parents and reproduce
        new_pop = []
        new_pop = selection.eliteism(population,fitness_list,elites)

        population = selection.roulette(population,fitness_list)
        # reproduce
        population = reproduce.crossover(population,300,0.6)
        population = reproduce.mutate(population,0.001)
        new_pop = new_pop + population
        new_pop = new_pop[:500]
    output.output("results.txt",population[best_idx])


if __name__ == "__main__":
    main()
