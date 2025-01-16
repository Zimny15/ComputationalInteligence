import numpy as np
import pandas as pd
import os
import sys
import random
import time


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
    return 1.0/path_length(path, dist_matrix) #Im krótsza trasa, tym większa wartość

def nn(dane): #Posłuży nam do wygenerowania początkowej populacji

    MAX = sys.maxsize    
    population = []  #Przechowywanie wszystkich wygenerowanych tras

    for start in range(len(dane)):  #Testujemy różne startowe miasta
        visited_cities = [start]

        while len(visited_cities) < len(dane):
            mask = dane[visited_cities[-1]].copy() 
            for i in visited_cities:
                mask[i] = MAX  #Zapobiegamy wybieraniu odwiedzonych miast
            next_city = np.argmin(mask)
            visited_cities.append(int(next_city))

        population.append(visited_cities) 

    return population #Zwracamy listę list z indeksami ścieżek 

def random_population(dane):
    size = len(dane)*2
    population = [random.sample(range(len(dane)), len(dane)) for _ in range(size)]
    return population

def roulette(init_pop, dist_matrix):
    
    pop_fit = []
    for i in range(len(init_pop)):
        pop_fit.append(fitness(init_pop[i], dist_matrix))

    probs = [pop_fit[i]/sum(pop_fit) for i in range(len(pop_fit))]
    
    #Wybór losowego punktu na kole ruletki
    #pick = random.uniform(0, sum(probs))
    #current = 0
    #Losujemy rodziców według obliczonych prawdopodobieństw
    parents = random.choices(init_pop, weights = probs, k = len(init_pop))  

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

def mutation(child, mutation_prob): #W przpyadku wylosowania mutacji zamieniamy dwa miasta miejscami, a potem znowu losujemy odwrócenie kolejności
    if mutation_prob > random.random():
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    if mutation_prob > random.random():
        x, y = random.sample(range(len(child)), 2)
        child[x:y+1] = reversed(child[x:y+1])
    return child

def order_crossover(parent1, parent2):
    #Losujemy dwa indeksy, między którymi będziemy wymieniać fragmenty
    n = len(parent1)
    point1, point2 = sorted(random.sample(range(n), 2))  #Sortujemy, żeby prawidłowo wybierać przedziały 
    #Tworzymy dziecko kopiując część rodzica 1
    child = [-1] * n
    child[point1:point2+1] = parent1[point1:point2+1]
    
    #Wypełniamy resztę dziecka, kopiując elementy z rodzica 2
    empty_positions = [i for i in range(n) if child[i] == -1]  #Lista indeksów do wypełnienia rodzicem 2
    insert_index = 0  #Służy do poruszania się po liście z indeksami 
    
    for city in parent2:
        if city not in child:
            child[empty_positions[insert_index]] = city
            insert_index += 1  #Przechodzimy do kolejnej pustej pozycji

    return child

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    point1, point2 = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[point1:point2+1] = parent1[point1:point2+1]

    for i in range(point1, point2+1):
        if parent2[i] not in child: #Znaleźliśmy gen, którego nie ma w dziecku
            swap_index = i #Miejsce jest już zajęte przez gen z rodzica1
            while child[swap_index] != -1: 
                swap_index = parent2.index(parent1[swap_index]) #Szukamy na jakim miejscu znajduje się w rodzicu2, gen z rodzica1, który zajmuje miejsce
            child[swap_index] = parent2[i]

    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]

    return child

def merge_crossover(parent1, parent2):
    child = []
    for p1, p2 in zip(parent1, parent2): #Jednocześnie iterujemy po obydwóch rodzicach 
        if p1 not in child:
            child.append(p1)
        if p2 not in child:
            child.append(p2)
    return child

def genetic_algorithm(init_method, dist_matrix, selection, crossover, elitism = True, max_gen=100, mutation_prob=0.01):

    time_start = time.time()
    elite_count = (len(dist_matrix)*5) // 100 #Chcemy, żeby elity stanowiły ok. 5% populacji 

    for gen in range(max_gen + 1):
        if gen == 0:
            init_pop = init_method(dist_matrix)  #Inicjalizacja populacji 

        #Gdy elitism = true: Zachowujemy najlepszego osobnika dla przyszłej populacji 
        if elitism:
            the_best = sorted(init_pop, key=lambda i: -fitness(i, dist_matrix))[:elite_count]
            init_pop = sorted(init_pop, key=lambda i: -fitness(i, dist_matrix))[elite_count:] #Będziemy krzyżowali osobniki bez elit
        else:
            the_best = []

        #Selekcja
        parents = selection(init_pop, dist_matrix)
        
        #Krzyżowanie
        next_pop = [] #Tutaj przechowujemy nowe pokolenie
        if len(parents) % 2 == 0:
            for i in range(0, len(parents), 2): #Dzieci zastępują rodziców i mutują
                child1 = crossover(parents[i], parents[i+1])
                child2 = crossover(parents[i], parents[i+1])

                child1 = mutation(child1, mutation_prob)
                child2 = mutation(child2, mutation_prob)

                next_pop.append(child1)
                next_pop.append(child2)
        else: #Przypadek gdy liczba miast jest nieparzysta
            for i in range(0, len(parents)-1, 2):
                child1 = crossover(parents[i], parents[i+1])
                child2 = crossover(parents[i], parents[i+1])

                child1 = mutation(child1, mutation_prob)
                child2 = mutation(child2, mutation_prob)

                next_pop.append(child1)
                next_pop.append(child2)
            last_child = parents[-1]
            last_child = mutation(last_child, mutation_prob)
            next_pop.append(last_child)

        init_pop = next_pop + the_best #Zastępujemy starą generacje nowym pokoleniem

    best_path = max(init_pop, key=lambda i: fitness(i, dist_matrix))
    total_road = path_length(best_path, dist_matrix)
    best_path = [(city + 1) for city in best_path]
    time_end = time.time()
    ex_time = time_end - time_start
    return best_path, total_road, ex_time


## Zapistywanie ##

data_source = [(dane1, "TSP_48"), (dane2, "TSP_76"), (dane3, "TSP_127")]
variables = selection_methods = [("Tournament", tournament), ("Linear rank",linear_rank),("Roulette", roulette)]
#variables =crossover_methods = [("OX1",order_crossover), ("PMX",pmx_crossover), ("MX",merge_crossover)]
#variables =mutation_probs = [("1%",0.01),("10%",0.1),("50%",0.5),("70%",0.7)]
#variables = gen_num = [("500",500),("1000", 1000),("5000",5000),("10000", 10000)]
results = {}

for data, dataset_name in data_source:
    print("Data set: ", dataset_name)
    for method_name, method in variables:
        best_path = []
        best_road = sys.maxsize
        total_road = 0
        avg_road = 0
        total_time = 0
        print("Metoda: ", method_name)
        for _ in range(10):
            print("Iteracja nr: ", _)
            path, road, ex_time = genetic_algorithm(nn, data, selection=method, crossover=pmx_crossover, max_gen=1000, mutation_prob=0.1)
            
            if road < best_road:
                best_road = road
                best_path = path
            
            total_road += road
            total_time += ex_time  

        avg_road = total_road / 10
        avg_time = total_time / 10

        results.setdefault(dataset_name, []).append({
            "Metoda selekcji": method_name,
            "Najlepsza droga": best_road,
            "Najlepsza trasa": str(best_path),
            "Średnia długość trasy": avg_road,
            "Średni czas (s)": avg_time
        })

# Tworzenie pliku Excel i zapis wyników
excel_writer = pd.ExcelWriter("wyniki_TSP_gen_selection.xlsx", engine="xlsxwriter")

for dataset_name, dataset_results in results.items():
    df = pd.DataFrame(dataset_results)
    df.to_excel(excel_writer, sheet_name=dataset_name, index=False)

excel_writer.close()

print("Zapisywanie skończone")
