from fitness import fitness
import inital_pop as pop
import selection
import reproduce
import output

def main():
    generations:int =  2000
    gait_length:int = 40
    # huge search space
    max_pop:int = 4000
    population = pop.gen_population(max_pop, gait_length)
    fit = fitness(gait_length)
    for gen in range(generations):
        fitness_list = []
        # generate fitness list
        best_idx = 0
        for individual in population:
            fitness_list.append(fit.get_fitness(individual))
        best_idx = fitness_list.index(max(fitness_list))
        print("generation:",gen,"| best index: ",best_idx, "| fitness: ",fitness_list[best_idx])

        # select new parents and reproduce
        population = selection.tournament(population,fitness_list,5)
        # population = selection.roulette(population,fitness_list)

        # reproduce
        # population = reproduce.crossover(population,gait_length,0.6)
        population = reproduce.uniform_crossover(population,gait_length,0.7)
        population = reproduce.mutate(population,0.02)
    output.output("results.txt",population[best_idx])


if __name__ == "__main__":
    main()
