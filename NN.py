import pandas as pd 
import numpy as np
import os 
import sys
import time

wd = os.path.dirname(os.path.realpath(__file__))
dane1 = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane2 = pd.read_excel(wd+'/dane/Dane_TSP_76.xlsx', index_col= 0, header = 0)
dane3 = pd.read_excel(wd+'/dane/Dane_TSP_127.xlsx', index_col= 0, header = 0)
dane1 = np.array(dane1)
dane2 = np.array(dane2)
dane3 = np.array(dane3)

np.fill_diagonal(dane2,0) #Plik _76 nie zawierał 0 po przekątnych

MAX = sys.maxsize

def path_length(route, dist_matrix):
    length = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    length += dist_matrix[route[-1], route[0]]  #Powrót do miasta startowego
    return length

def nn(dane, save_name):
    start_time = time.time()

    all_paths = [] #Ta lista ma służyć do zapisania wyników w xlsx (Na potrzeby sprawozdania)
    all_scores = [] #Wyniki funkcji celu do zapisanych ścieżek 

    total_road = MAX  #Przechowywanie najlepszej trasy
    best_path = []  #Przechowywanie najlepszej ścieżki

    for start in range(len(dane)):  #Testujemy różne startowe miasta
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy() 
            for i in visited_cities:
                mask[i] = MAX #Zapobiegam wybieraniu odwiedzonych miast

            next_city = np.argmin(mask)
            visited_cities.append(next_city)

        #Obliczenie długości trasy
        road_length = path_length(visited_cities, dane)

        all_paths.append(visited_cities)
        all_scores.append(road_length)
        
        #Sprawdzenie, czy otrzymaliśmy lepsze rozwiązanie
        if road_length < total_road:
            total_road = road_length
            best_path = visited_cities.copy()

    end_time = time.time()  
    execution_time = end_time - start_time  #Czas wykonania

    print("THE BEST \n")
    print("Najlepsza trasa:", [int(city + 1) for city in best_path])
    print("Najkrótsza długość trasy:", total_road)

    for i in range(len(all_paths)):
        all_paths[i] = [city + 1 for city in all_paths[i]]

    df = pd.DataFrame(all_paths).T  
    df.columns = [f"{i+1}" for i in range(len(all_paths))]  #Nazwy kolumn
    #Dodanie wartości `all_scores` jako ostatniego wiersza
    df.loc["Długość trasy"] = all_scores

    best_result_df = pd.DataFrame({
        "Najlepsza trasa": [", ".join(str(city + 1) for city in best_path)],
        "Najkrótsza długość trasy": [total_road],
        "Czas wykonania (s)": [execution_time]
    })

    output_path = wd + "/dane/" + save_name + ".xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Wszystkie trasy", index=False)
        best_result_df.to_excel(writer, sheet_name="Najlepszy wynik", index=False)

nn(dane3, "wyniki_TSP_NN_127")
