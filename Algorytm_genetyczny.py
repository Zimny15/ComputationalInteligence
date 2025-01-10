import numpy as np
import pandas as pd
import os
import random

wd = os.path.dirname(os.path.realpath(__file__))
dane1 = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane2 = pd.read_excel(wd+'/dane/Dane_TSP_76.xlsx', index_col= 0, header = 0)
dane3 = pd.read_excel(wd+'/dane/Dane_TSP_127.xlsx', index_col= 0, header = 0)
dane1 = np.array(dane1)
dane2 = np.array(dane2)
dane3 = np.array(dane3)

np.fill_diagonal(dane2,0) #Plik _76 nie zawierał 0 po przekątnych

def path_length(path, dist_matrix):
    length = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    length += dist_matrix[path[-1], path[0]]  #Powrót do miasta startowego
    return length

def fitness(path, dist_matrix):
    return 1/path_length(path, dist_matrix) #Im krótsza trasa, tym większa wartość

def nn(dane): #Posłuży nam do wygenerowania początkowej populacji

    population = []  #Przechowywanie wszystkich wygenerowanych tras

    for start in range(len(dane)):  #Testujemy różne startowe miasta
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy() 
            for i in visited_cities:
                mask[i] = max(mask)  #Zapobiegamy wybieraniu odwiedzonych miast
            next_city = np.argmin(mask)
            visited_cities.append(int(next_city))

        population.append(visited_cities) 

    return population #Zwracamy listę list z indeksami ścieżek 

## DODAJ Co najmniej dwie metody selekcji i dwie metody krzyżowania ##

def roulette(init_pop, dist_matrix):

    pop_fit = []
    for _ in range(len(init_pop)):
        pop_fit.append(fitness(init_pop[_], dist_matrix))

    probs = [pop_fit[i]/sum(pop_fit) for i in range(len(pop_fit))]

    #Losujemy rodziców według obliczonych prawdopodobieństw
    parents = random.choices(init_pop, weights = probs, k = len(init_pop)) #Losujemy rodziców 

    return parents

def linear_rank(init_pop, dist_matrix, sp=1.5): #sp - selection pressure

    fitness_values = [fitness(i,dist_matrix) for i in init_pop]
    
    #Sortujemy populację według fitness (od najsłabszego do najlepszego)
    sorted_pop = [x for _, x in sorted(zip(fitness_values, init_pop))]
    
    N = len(init_pop)  #Liczba osobników
    
    #Nadajemy rangi (najgorszy osobnik ma rangę 1, najlepszy rangę N)
    ranks = list(range(1, N + 1))
    
    #Obliczamy prawdopodobieństwa wyboru na podstawie rangi
    probs = [(sp/N) + ((2*(sp-1)*(r-1)) / (N*(N-1))) for r in ranks]

    #Losujemy nowych rodziców według obliczonych prawdopodobieństw
    parents = random.choices(sorted_pop, weights=probs, k=N)

    return parents

def tournament(init_pop, dist_matrix, k=3):
    parents = []
    for _ in range(len(init_pop)):
        players = random.sample(init_pop, k=k) #Losujemy k oponentów do turnieju 
        winner = max(players, key=lambda i: fitness(i, dist_matrix)) #Zwycięzcą jest ten o największym dopasowaniu
        parents.append(winner) #Dodajemy zwycięzce do listy przyszych rodziców 

    return parents

def mutation(child, mutation_prob): #W przpyadku wylosowania mutacji zamieniamy dwa miasta miejscami
    n = len(child)
    if mutation_prob > random.random():
        i,j = random.sample(range(0, n), 2)
        temp = child[i]
        child[i] = child[j]
        child[j] = temp
    return child

def order_crossover(parent1, parent2):
    #Losujemy dwa indeksy, między którymi będziemy wymieniać fragmenty
    n = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(n), 2))  #Sortujemy, żeby prawidłowo wybierać przedziały 
    
    #Tworzymy dziecko kopiując część rodzica 1
    child = [None] * n
    child[cxpoint1:cxpoint2+1] = parent1[cxpoint1:cxpoint2+1]
    
    #Wypełniamy resztę dziecka, kopiując elementy z rodzica 2, pomijając te, które już są w dziecku
    current_pos = (cxpoint2 + 1) % n
    for i in range(n):
        item = parent2[(cxpoint2 + 1 + i) % n]
        if item not in child:
            child[current_pos] = item
            current_pos = (current_pos + 1) % n

    return child

def genetic_algorithm(dist_matrix, selection, elitism = True, max_gen=100, mutation_prob=0.01):

    for gen in range(max_gen + 1):

        if gen == 0:
            init_pop = nn(dist_matrix)  #Inicjalizacja populacji 
        
        #Selekcja
        parents = selection(init_pop, dist_matrix)

        #Krzyżowanie
        next_pop = [] #Tutaj przechowujemy nowe pokolenie
        if len(parents) % 2 == 0:
            for i in range(0, len(parents), 2): #Dzieci zastępują rodziców i mutują
                child1 = order_crossover(parents[i], parents[i+1])
                child2 = order_crossover(parents[i], parents[i+1])

                child1 = mutation(child1, mutation_prob)
                child2 = mutation(child2, mutation_prob)

                next_pop.append(child1)
                next_pop.append(child2)
        else: #Przypadek gdy liczba miast jest nieparzysta
            for i in range(0, len(parents)-1, 2):
                child1 = order_crossover(parents[i], parents[i+1])
                child2 = order_crossover(parents[i], parents[i+1])

                child1 = mutation(child1, mutation_prob)
                child2 = mutation(child2, mutation_prob)

                next_pop.append(child1)
                next_pop.append(child2)
            last_child = parents[-1]
            last_child = mutation(last_child, mutation_prob)
            next_pop.append(last_child)

        #Gdy elitism = true: Zachowujemy najlepszego osobnika dla przyszłej populacji 
        if elitism:
            the_best = max(init_pop, key=lambda i: fitness(i, dist_matrix))  
            next_pop[0] = the_best

        init_pop = next_pop #Zastępujemy starą generacje nowym pokoleniem
        
        #Najlepszy osobnik w obecnej generacji
        best_path = max(init_pop, key=lambda i: fitness(i, dist_matrix))
        total_road = path_length(best_path, dist_matrix)
        
        print("Generacja:", gen, "\nNajlepsze rozwiązanie:", total_road, [(city + 1) for city in best_path])


genetic_algorithm(dane1, max_gen=5000, selection= roulette)

