import pandas as pd 
import numpy as np
import os 

wd = os.path.dirname(os.path.realpath(__file__))
dane = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane = np.array(dane)

print(dane)

total_road = len(dane)*max(dane[0]) 

for start in range(len(dane)):  #Testujemy różne startowe miasta
    visited_cities = [start]
    road_length = 0

    while len(visited_cities) < len(dane):
        mask = dane[visited_cities[-1]].copy()
        for i in visited_cities:
            mask[i] = max(mask)  #Zapobiegam wybieraniu odwiedzonych miast

        next_city = np.argmin(mask)
        visited_cities.append(next_city)
        road_length += mask[next_city]

    #Powrót do miasta startowego
    road_length += dane[visited_cities[-1]][visited_cities[0]]
    for city in visited_cities:
        print(city + 1)  
    print(total_road)
    #Sprawdzenie, czy mamy krótszą trasę
    if road_length < total_road:
        total_road = road_length
        best_path = visited_cities

print("THE BEST\n")
for city in best_path:
    print(city + 1)  #Indeksy zaczynają się od 0, więc dodajemy 1
print(total_road)

