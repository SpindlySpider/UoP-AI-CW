from inital_pop import Population_obj
from fitness import fitness
import selection
import reproduce

def main():
    generations:int = 500
    population = Population_obj(10).gen_population()
    print(len(population[0][0]))
    fit = fitness()
    best_inx = 0
    for _ in range(generations):
        fitness_list = []
        # generate fitness list
        fit_val = 0
        for idx,individual in enumerate(population):
            fit_val:float = fit.leg_angles(individual)
            fitness_list.append(fit_val)
        best_idx = fitness_list.index(max(fitness_list))
        print(population[best_idx])
        print(best_idx, fitness_list[best_idx])
        # select new parents and reproduce
        population = selection.roulette(population,fitness_list)
        # reproduce
        population = reproduce.crossover(population,300,0.6)
        population = reproduce.mutate(population,0.025)
    file = open("results.txt","w")
    file.writelines(f"{population[best_idx]}")


if __name__ == "__main__":
    main()
