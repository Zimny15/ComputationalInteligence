import pandas as pd
import numpy as np

nazwa_pliku = "C:\\Users\\zimak\\Desktop\\IiE\\IO\\TSP_29.xlsx"
macierz = pd.read_excel(nazwa_pliku, index_col=0, header=0)
liczba_wierszy, liczba_kolumn = macierz.shape
liczba_miast = macierz.shape[0]
macierz_numpy = macierz.to_numpy()

if(liczba_kolumn != liczba_wierszy):
    print("Zła macierz odleglosci")
    exit()

najlepszy_dystans = float('inf')
najlepsza_trasa = []


for start in range(liczba_miast):
    odwiedzone = [start]
    dystans_calkowity = 0
    aktualne_miasto = start

    while(len(odwiedzone) < liczba_miast):
        min_dystans = float('inf')
        nastepne_miasto = None

        for miasto in range(liczba_miast):
            if miasto not in odwiedzone and macierz_numpy[aktualne_miasto, miasto] < min_dystans:
                min_dystans = macierz_numpy[aktualne_miasto, miasto]
                nastepne_miasto = miasto

        odwiedzone.append(nastepne_miasto)
        dystans_calkowity += min_dystans
        aktualne_miasto = nastepne_miasto
    
    dystans_calkowity += macierz_numpy[aktualne_miasto, start]
    odwiedzone.append(start)

    if(dystans_calkowity < najlepszy_dystans):
        najlepszy_dystans = dystans_calkowity
        najlepsza_trasa = odwiedzone

    print(f"Trasa: {', '.join(map(str, [x + 1 for x in odwiedzone]))}")
    print(f"Dystans całkowity: {dystans_calkowity}\n")


    # Wypisanie najlepszej trasy
print("Najlepsza trasa:")
print(f"Trasa: {', '.join(map(str, [x + 1 for x in najlepsza_trasa]))}")
print(f"Dystans całkowity: {najlepszy_dystans}")