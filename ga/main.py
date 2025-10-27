from inital_pop import Population_obj
from fitness import fitness
import selection
import reproduce
import output

def main():
    generations:int =  1350
    gait_length = 30
    max_pop = 1000
    population = Population_obj(gait_length,max_pop).gen_population()
    print(len(population[0][0]))
    fit = fitness(gait_length)
    best_inx = 0
    for gen in range(generations):
        fitness_list = []
        # generate fitness list
        fit_val = 0
        for idx,individual in enumerate(population):
            fit_val:float = fit.get_fitness(individual)
            fitness_list.append(fit_val)
        best_idx = fitness_list.index(max(fitness_list))
        print(population[best_idx])
        print(gen,best_idx, fitness_list[best_idx])

        # select new parents and reproduce

        population = selection.tournament(population,fitness_list,3)
        # population = selection.roulette(population,fitness_list)

        # reproduce
        # population = reproduce.crossover(population,gait_length,0.6)
        population = reproduce.uniform_crossover(population,gait_length,0.7)
        population = reproduce.mutate(population,0.02)
    output.output("results.txt",population[best_idx])


if __name__ == "__main__":
    main()
