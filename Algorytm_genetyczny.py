import numpy as np
import pandas as pd
import os
import random

wd = os.getcwd()
dane = pd.read_excel(wd + '/dane/Dane_TSP_48.xlsx', index_col=0, header=0)
dane = np.array(dane)

#Funkcja obliczająca długość trasy
def path_length(route, dist_matrix):
    length = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    length += dist_matrix[route[-1], route[0]]  # Powrót do miasta startowego
    return length

#Algorytm najbliższego sąsiada - inicjalizacja populacji
def nn(dane):
    population = {}

    for start in range(len(dane)):  
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy()
            for i in visited_cities:
                mask[i] = max(mask)  #Zapobiegam wybieraniu odwiedzonych miast

            next_city = np.argmin(mask)
            visited_cities.append(next_city)

        visited_cities.append(start)  #Powrót do startu
        for i in range(len(visited_cities)):
            visited_cities[i] = int(visited_cities[i])
        road_length = path_length(visited_cities, dane)
        population[tuple(visited_cities)] = int(road_length)

    return population

#Funkcja selekcji turniejowej (wybiera najlepsze rozwiązania)
def tournament_selection(population, k=3):
    candidates = random.sample(list(population.keys()), k)
    best = min(candidates, key=lambda route: population[route])
    return best

#Krzyżowanie OX (Ordered Crossover)
def ordered_crossover(parent1, parent2):
    size = len(parent1) - 1  #Ostatni element to powrót do startu
    start, end = sorted(random.sample(range(1, size), 2))  #Nie zmieniamy miasta startowego

    child = [-1] * (size + 1)
    child[start:end] = parent1[start:end]

    #Wypełnianie brakujących miast z parent2
    idx = end
    for gene in parent2[1:size]:  #Pomijamy miasto startowe
        if gene not in child:
            if idx >= size:
                idx = 1  #Pomijamy startowe miasto
            child[idx] = gene
            idx += 1

    child[-1] = child[0]  #Powrót do miasta startowego
    return tuple(child)

#Mutacja swap (zamienia miejscami dwa losowe miasta)
def swap_mutation(route):
    route = list(route)
    size = len(route) - 1
    a, b = random.sample(range(1, size), 2)  #Nie zmieniamy miasta startowego
    route[a], route[b] = route[b], route[a]
    return tuple(route)

#Algorytm genetyczny
def genetic_algorithm(dane, max_gen=100, pop_size=30, crossover_rate=0.8, mutation_rate=0.2):
    #Inicjalizacja populacji
    population = nn(dane)

    for gen in range(1, max_gen + 1):
        new_population = {}

        while len(new_population) < pop_size:
            #Selekcja rodziców
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            #Krzyżowanie
            if random.random() < crossover_rate:
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1  #Bez zmian

            #Mutacja
            if random.random() < mutation_rate:
                child = swap_mutation(child)

            #Obliczenie długości trasy dziecka
            road_length = path_length(child, dane)
            new_population[child] = road_length

        #Aktualizacja populacji
        population = new_population

        #Wyświetlenie najlepszej trasy w generacji
        best_route = min(population, key=population.get)
        print(f"Generacja {gen}: Najlepsza długość trasy = {population[best_route]}")

    return min(population, key=population.get), population[min(population, key=population.get)]

best_route, best_length = genetic_algorithm(dane, max_gen=100, pop_size=30)
print("\nNajlepsza trasa:", best_route)
print("Najkrótsza długość trasy:", best_length)

