import pandas as pd 
import numpy as np
import random
import os 

wd = os.path.dirname(os.path.realpath(__file__))
dane = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane = np.array(dane)

print(dane)

def path_length(route, dist_matrix):
    length = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    length += dist_matrix[route[-1], route[0]]  #Powrót do miasta startowego
    return length

def swap(path):
    new_path = path.copy()
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def two_opt(path):
    new_path = path.copy()
    i, j = sorted(random.sample(range(len(path)), 2))
    new_path[i:j+1] = reversed(new_path[i:j+1])
    return new_path

def insert(path):
    new_path = path.copy()
    i, j = random.sample(range(len(new_path)), 2)
    city = new_path.pop(i)
    new_path.insert(j, city)
    return new_path

### WERSJA NA CZYSTO ###

#Za argument 'method' należy podstawić - swap/two_opt/insert - aby spróbować ulepszyć rozwiązanie

def nn(dane, method=None):

    total_road = len(dane) * max(dane[0])  #Przechowywanie najlepszej trasy
    best_path = []  #Przechowywanie najlepszej ścieżki

    for start in range(len(dane)):  #Testujemy różne startowe miasta
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy() 
            for i in visited_cities:
                mask[i] = max(mask)  #Zapobiegam wybieraniu odwiedzonych miast

            next_city = np.argmin(mask)
            visited_cities.append(next_city)

        #Powrót do miasta startowego
        visited_cities.append(start)

        #Obliczenie długości trasy
        road_length = path_length(visited_cities, dane)

        #Ulepszanie rozwiązania
        if method is not None:
            for i in range(5000):
                new_path = method(visited_cities)
                new_length = path_length(new_path, dane)

                if new_length < road_length:
                    visited_cities = new_path.copy()
                    road_length = new_length

        print("Aktualna trasa:", [int(city + 1) for city in visited_cities])
        print("Długość trasy:", road_length, "\n")

        #Sprawdzenie, czy otrzymaliśmy lepsze rozwiązanie
        if road_length < total_road:
            total_road = road_length
            best_path = visited_cities.copy()

    print("THE BEST \n")
    print("Najlepsza trasa:", [int(city + 1) for city in best_path])
    print("Najkrótsza długość trasy:", total_road)

nn(dane, insert)