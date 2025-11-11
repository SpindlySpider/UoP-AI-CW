from fitness import fitness
from fitness_graph import plot_fitness_graph
import inital_pop as pop
import selection
import reproduce
import output


def main():
    generations:int =  600
    gait_length:int = 40
    max_pop:int = 3000
    population = pop.gen_population(max_pop, gait_length)
    fit = fitness(gait_length)
    fitness_over_time = []
    avg_fitness_over_time = []
    for gen in range(generations):
        fitness_list = []
        # generate fitness list
        best_idx = 0
        for individual in population:
            fitness_list.append(fit.get_fitness(individual))
        best_idx = fitness_list.index(max(fitness_list))
        print("generation:",gen,"| best index: ",best_idx, "| fitness: ",fitness_list[best_idx])
        avg_fitness = sum(fitness_list) / len(fitness_list)
        fitness_over_time.append(fitness_list[best_idx])
        avg_fitness_over_time.append(avg_fitness)
        


        # select new parents and reproduce
        population = selection.tournament(population,fitness_list,10)
        # population = selection.roulette(population,fitness_list)

        # reproduce
        # population = reproduce.crossover(population,gait_length,0.6)
        population = reproduce.uniform_crossover(population,gait_length,0.7)
        population = reproduce.mutate(population,0.025)
    output.output_gait("results.txt",population[best_idx],300)


    plot_fitness_graph(fitness_over_time, avg_fitness_over_time, generations)


if __name__ == "__main__":
    main()
