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

def nn(dane): #Posłuży nam do wygenerowania początkowej populacji

    population = []  #Przechowywanie wszystkich wygenerowanych tras

    for start in range(len(dane)):  #Testujemy różne startowe miasta
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy() 
            for i in visited_cities:
                mask[i] = max(mask)  #Zapobiegam wybieraniu odwiedzonych miast
            next_city = np.argmin(mask)
            visited_cities.append(int(next_city))

        population.append(visited_cities) 

    return population #Zwracamy listę list z indeksami ścieżek 

## DODAJ Co najmniej dwie metody selekcji i dwie metody krzyżowania ##

def roulette(init_pop, dist_matrix):
    tot_path = 0
    probs = []  # Prawdopodobieństwo bycia wybranym
    
    # Obliczanie całkowitej długości ścieżek
    for i in range(len(init_pop)):
        tot_path += path_length(init_pop[i], dist_matrix)

    # Obliczanie prawdopodobieństw
    for i in range(len(init_pop)):
        probs.append(1 - (path_length(init_pop[i], dist_matrix) / tot_path))  # Dodajemy prawdopodobieństwo
    
    # Normalizacja prawdopodobieństw
    total_prob = sum(probs)
    normalized_probs = [p / total_prob for p in probs]
    
    # Skumulowane prawdopodobieństwa
    cumulative_probs = [sum(normalized_probs[:i+1]) for i in range(len(normalized_probs))]
    
    selected = []
    for j in range(int(len(init_pop) * 3 / 5)):
        r = random.random()  # Losujemy liczbę z przedziału [0, 1)
        
        # Wybieramy element na podstawie skumulowanego prawdopodobieństwa
        for i, cumulative_prob in enumerate(cumulative_probs):
            if r <= cumulative_prob:
                selected.append(init_pop[i])  # Dodajemy do listy wybranych
                break
    
    return selected

def genetic_algorithm(dist_matrix, max_gen=100, mutation_prob=0.01):
    init_pop = nn(dist_matrix)  # Inicjalizacja populacji (zakładając, że nn() jest zdefiniowane)

    # Selekcja
    selected = roulette(init_pop, dist_matrix)

    print(selected)
    #Metoda turnieju 
    #Potencjalnie elitist

    #Krzyżowanie

    #Mutacja



