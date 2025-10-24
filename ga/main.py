from inital_pop import Population_obj
from fitness import fitness
import selection
import reproduce

def main():
    generations:int = 500
    population = Population_obj(10).gen_population()
    fit = fitness()
    best_inx = 0
    for _ in range(generations):
        fitness_list = []
        # generate fitness list
        for idx,individual in enumerate(population):
            # print(fit.leg_angles(individual))
            fit_val = fit.leg_angles(individual)
            # print("pop1:",population[0][0])
            # print("pop2:",population[1][0])
            # print("appending:",population[idx][0],fit_val)

            fitness_list.append(fit_val)
        # print(fitness_list)
        best_idx:int = fitness_list.index(max(fitness_list))
        # print(fitness_list)
        print(population[best_idx])
        print(best_idx,fit_val)
        # select new parents and reproduce
        population = selection.tournament(population,fitness_list)
        # reproduce
        population = reproduce.crossover(population,300)
        population = reproduce.mutate(population,0.002)
    file = open("results.txt","w")
    file.writelines(f"{population[best_idx]}")


if __name__ == "__main__":
    main()
