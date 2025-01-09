import pandas as pd 
import numpy as np
import os 

wd = os.path.dirname(os.path.realpath(__file__))
dane1 = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane2 = pd.read_excel(wd+'/dane/Dane_TSP_76.xlsx', index_col= 0, header = 0)
dane3 = pd.read_excel(wd+'/dane/Dane_TSP_127.xlsx', index_col= 0, header = 0)
dane1 = np.array(dane1)
dane2 = np.array(dane2)
np.fill_diagonal(dane2,0) 
dane3 = np.array(dane3)

def path_length(route, dist_matrix):
    length = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    length += dist_matrix[route[-1], route[0]]  #Powrót do miasta startowego
    return length

def nn(dane):

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

        print("Aktualna trasa:", [int(city + 1) for city in visited_cities])
        print("Długość trasy:", road_length, "\n")

        #Sprawdzenie, czy otrzymaliśmy lepsze rozwiązanie
        if road_length < total_road:
            total_road = road_length
            best_path = visited_cities.copy()

    print("THE BEST \n")
    print("Najlepsza trasa:", [int(city + 1) for city in best_path])
    print("Najkrótsza długość trasy:", total_road)

nn(dane2)
