import pandas as pd 
import numpy as np
import random
import os 

wd = os.path.dirname(os.path.realpath(__file__))
dane = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col= 0, header = 0)
dane = np.array(dane)

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


# Symulowane wyżarzanie
def symulowane_wyżarzanie(macierz, epoki, temperatura_poczatkowa, schladzanie, liczba_prob, metoda_ruchu):
    liczba_miast = len(macierz)

    # Inicjalizacja
    aktualna_trasa = list(range(liczba_miast))
    random.shuffle(aktualna_trasa)
    najlepsza_trasa = aktualna_trasa.copy()
    najlepszy_dystans = path_length(najlepsza_trasa, macierz)

    temperatura = temperatura_poczatkowa

    for epoka in range(epoki):
        for _ in range(liczba_prob):
            # Generowanie nowej trasy według wybranej metody ruchu
            nowa_trasa = metoda_ruchu(aktualna_trasa)
            aktualny_dystans = path_length(aktualna_trasa, macierz)
            nowy_dystans = path_length(nowa_trasa, macierz)

            # Kryterium akceptacji nowego rozwiązania
            if nowy_dystans < aktualny_dystans:
                aktualna_trasa = nowa_trasa
            else:
                delta = nowy_dystans - aktualny_dystans
                prawdopodobienstwo = np.exp(-delta / temperatura)
                if random.random() < prawdopodobienstwo:
                    aktualna_trasa = nowa_trasa

            # Aktualizacja najlepszego rozwiązania
            if path_length(aktualna_trasa, macierz) < najlepszy_dystans:
                najlepsza_trasa = aktualna_trasa.copy()
                najlepszy_dystans = path_length(najlepsza_trasa, macierz)

        # Obniżenie temperatury
        temperatura *= schladzanie
        print(f"Epoka {epoka + 1}/{epoki}: Najlepszy dystans = {najlepszy_dystans}")

    return najlepsza_trasa, najlepszy_dystans

# Wybór metody ruchu (możesz podać swap, two_opt lub insert)
metoda_ruchu = two_opt  # Możesz zmienić na swap lub insert





# Przykład parametrów
epoki = 1000
temperatura_poczatkowa = 10000
schladzanie = 0.99
liczba_prob = 1000


# Lista do przechowywania wyników
wszystkie_rozwiazania = []

for i in range(100):
    najlepsza_trasa, najlepszy_dystans = symulowane_wyżarzanie(
        dane, epoki, temperatura_poczatkowa, schladzanie, liczba_prob, metoda_ruchu
    )
    najlepsza_trasa_od_1 = list(map(lambda x: x + 1, najlepsza_trasa)) 
    najlepsza_trasa_od_1.append(najlepsza_trasa_od_1[0])
    
    # Przechowywanie wyników
    wszystkie_rozwiazania.append((najlepszy_dystans, najlepsza_trasa_od_1))
    
    # Wyświetlenie wyniku z bieżącej iteracji
    print(f"Iteracja {i + 1}:")
    print(f"  Najlepsza trasa: {najlepsza_trasa_od_1}")
    print(f"  Najlepszy dystans: {najlepszy_dystans}\n")

# Znalezienie najlepszego z najlepszych rozwiązań
globalnie_najlepszy_dystans, globalnie_najlepsza_trasa = min(wszystkie_rozwiazania, key=lambda x: x[0])

# Wyświetlenie globalnie najlepszego rozwiązania
print("Najlepsze rozwiązanie z 100 iteracji:")
print(f"  Najlepsza trasa: {globalnie_najlepsza_trasa}")
print(f"  Najlepszy dystans: {globalnie_najlepszy_dystans}")

