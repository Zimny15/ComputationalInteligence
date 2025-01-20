import pandas as pd 
import numpy as np
import random
import os
import time
from datetime import datetime

# Pobranie ścieżki roboczej pliku wykonywanego
wd = os.path.dirname(os.path.realpath(__file__))

# Wczytanie danych z plików Excel do tablic NumPy
dane1 = pd.read_excel(wd+'/dane/Dane_TSP_48.xlsx', index_col=0, header=0)
dane2 = pd.read_excel(wd+'/dane/Dane_TSP_76.xlsx', index_col=0, header=0)
dane3 = pd.read_excel(wd+'/dane/Dane_TSP_127.xlsx', index_col=0, header=0)

# Konwersja danych do macierzy NumPy
dane1 = np.array(dane1)
dane2 = np.array(dane2)
np.fill_diagonal(dane2, 0)  # Zerowanie przekątnej (dystans z miasta do siebie samego)
dane3 = np.array(dane3)

# Funkcja obliczająca długość trasy na podstawie macierzy odległości
def path_length(route, dist_matrix):
    length = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    length += dist_matrix[route[-1], route[0]]  # Dodanie dystansu powrotu do miasta startowego
    return length

# Funkcja generująca nową trasę przez zamianę miejscami dwóch losowych miast
def swap(path):
    new_path = path.copy()
    i, j = random.sample(range(len(path)), 2)  # Losowanie dwóch różnych indeksów
    new_path[i], new_path[j] = new_path[j], new_path[i]  # Zamiana miast
    return new_path

# Funkcja generująca nową trasę przez odwrócenie fragmentu ścieżki (metoda Two-Opt)
def two_opt(path):
    new_path = path.copy()
    i, j = sorted(random.sample(range(len(path)), 2))  # Losowanie zakresu do odwrócenia
    new_path[i:j+1] = reversed(new_path[i:j+1])  # Odwrócenie fragmentu
    return new_path

# Funkcja generująca nową trasę przez wstawienie jednego miasta w inne miejsce
def insert(path):
    new_path = path.copy()
    i, j = random.sample(range(len(new_path)), 2)  # Losowanie dwóch różnych indeksów
    city = new_path.pop(i)  # Usunięcie miasta z jednego miejsca
    new_path.insert(j, city)  # Wstawienie go w inne miejsce
    return new_path

# Główna funkcja implementująca algorytm symulowanego wyżarzania
def symulowane_wyżarzanie(
        macierz, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
        liczba_prob=1000, liczba_iteracji=50, metoda_ruchu=insert, wd="."
    ):
    liczba_miast = len(macierz)  # Liczba miast do odwiedzenia
    folder_wyniki = os.path.join(wd, "wyniki_wyzarzanie")  # Ścieżka do zapisu wyników
    os.makedirs(folder_wyniki, exist_ok=True)  # Tworzenie folderu, jeśli nie istnieje

    wszystkie_rozwiazania = []  # Lista przechowująca wyniki dla każdej iteracji
    start_time = time.time()  # Rozpoczęcie pomiaru czasu działania algorytmu

    # Pętla dla wielu iteracji algorytmu (w celu znalezienia najlepszego rozwiązania globalnie)
    for i in range(liczba_iteracji):       
        # Inicjalizacja trasy i wartości początkowych
        aktualna_trasa = list(range(liczba_miast))  # Początkowa trasa (kolejność miast)
        random.shuffle(aktualna_trasa)  # Losowe permutowanie trasy
        najlepsza_trasa = aktualna_trasa.copy()  # Najlepsza trasa na danej iteracji
        najlepszy_dystans = path_length(najlepsza_trasa, macierz)  # Najlepszy dystans

        temperatura = temperatura_poczatkowa  # Ustawienie temperatury początkowej

        # Pętla dla epok (temperatura maleje w każdej epoce)
        for epoka in range(epoki):
            # Pętla dla prób w ramach jednej epoki
            for _ in range(liczba_prob):
                nowa_trasa = metoda_ruchu(aktualna_trasa)  # Generowanie nowej trasy
                aktualny_dystans = path_length(aktualna_trasa, macierz)  # Dystans aktualnej trasy
                nowy_dystans = path_length(nowa_trasa, macierz)  # Dystans nowej trasy

                # Kryterium akceptacji nowego rozwiązania
                if nowy_dystans < aktualny_dystans:  # Jeśli nowy dystans jest mniejszy, akceptujemy
                    aktualna_trasa = nowa_trasa
                else:
                    delta = nowy_dystans - aktualny_dystans  # Różnica między dystansami
                    prawdopodobienstwo = np.exp(-delta / temperatura)  # Obliczenie prawdopodobieństwa akceptacji
                    if random.random() < prawdopodobienstwo:  # Losowa decyzja o akceptacji
                        aktualna_trasa = nowa_trasa

                # Aktualizacja najlepszego rozwiązania w ramach iteracji
                if path_length(aktualna_trasa, macierz) < najlepszy_dystans:
                    najlepsza_trasa = aktualna_trasa.copy()
                    najlepszy_dystans = path_length(najlepsza_trasa, macierz)

            temperatura *= schladzanie  # Obniżenie temperatury (proces schładzania)

        # Zapisanie trasy w formacie od 1 (dla łatwiejszej interpretacji)
        najlepsza_trasa_od_1 = list(map(lambda x: x + 1, najlepsza_trasa))
        najlepsza_trasa_od_1.append(najlepsza_trasa_od_1[0])  # Dodanie powrotu do miasta startowego
        wszystkie_rozwiazania.append((najlepszy_dystans, najlepsza_trasa_od_1))

    end_time = time.time()  # Koniec pomiaru czasu działania algorytmu
    czas_dzialania = end_time - start_time  # Obliczenie czasu działania algorytmu

    # Obliczanie średniego dystansu
    srednia_wartosc_dystansu = np.mean([dystans for dystans, _ in wszystkie_rozwiazania])

    # Znalezienie najlepszego rozwiązania globalnego
    globalnie_najlepszy_dystans, globalnie_najlepsza_trasa = min(wszystkie_rozwiazania, key=lambda x: x[0])

    # Przygotowanie nazwy pliku wynikowego z dynamiczną datą i godziną
    nazwa_pliku = os.path.join(folder_wyniki, f"wyniki_symulacji_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Przygotowanie nagłówka z parametrami symulacji
    naglowek = (f"\nWyniki symulowanego wyżarzania\n"
                f"Metoda ruchu: {metoda_ruchu.__name__}\n"
                f"Liczba epok: {epoki}\n"
                f"Temperatura początkowa: {temperatura_poczatkowa}\n"
                f"Schładzanie: {schladzanie}\n"
                f"Liczba prób: {liczba_prob}\n"
                f"Liczba iteracji: {liczba_iteracji}\n\n")

    # Zapis wyników do pliku
    with open(nazwa_pliku, 'w') as plik:
        plik.write(naglowek)
        for i, (dystans, trasa) in enumerate(wszystkie_rozwiazania):
            plik.write(f"Iteracja {i + 1}:\n")
            plik.write(f"  Najlepsza trasa: {trasa}\n")
            plik.write(f"  Najlepszy dystans: {dystans}\n\n")
        plik.write(f"Najlepsze rozwiązanie z {liczba_iteracji} iteracji:\n")
        plik.write(f"  Najlepsza trasa: {globalnie_najlepsza_trasa}\n")
        plik.write(f"  Najlepszy dystans: {globalnie_najlepszy_dystans}\n")
        plik.write(f"Średnia wartość dystansu z {liczba_iteracji} iteracji: {srednia_wartosc_dystansu:.2f}\n")
        plik.write(f"Czas działania algorytmu: {czas_dzialania:.2f} sekund\n")

    # Informacja o zapisaniu wyników
    print(f"Wyniki zostały zapisane w pliku '{nazwa_pliku}'.")

    # Zwrot najlepszego rozwiązania
    return globalnie_najlepsza_trasa, globalnie_najlepszy_dystans 


# #Dane dla 48
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=1, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=50, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=100, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane1, epoki=10, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=25, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=100, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=1000, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.8,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.9,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.95,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")



# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=swap, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=insert, wd=".")



# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=1000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=5000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane1, epoki=800, temperatura_poczatkowa=20000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")


# #Dane dla 76


# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=1, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=50, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=100, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane2, epoki=10, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=25, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=100, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=1000, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.8,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.9,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.95,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")



# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=swap, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=insert, wd=".")



# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=1000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=5000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane2, epoki=800, temperatura_poczatkowa=20000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")


#dane 127
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=1, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=50, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=100, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane3, epoki=10, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=25, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=100, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=1000, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")


# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.8,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.9,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.95,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")



# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=swap, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=insert, wd=".")



# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=1000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=5000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=10000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")
# symulowane_wyżarzanie(dane3, epoki=800, temperatura_poczatkowa=20000, schladzanie=0.99,
#         liczba_prob=1000, liczba_iteracji=10, metoda_ruchu=two_opt, wd=".")