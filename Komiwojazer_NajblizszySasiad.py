import pandas as pd
import numpy as np

nazwa_pliku = "C:\\Users\\zimak\\Desktop\\IiE\\IO\\TSP_29.xlsx"
macierz = pd.read_excel(nazwa_pliku, index_col=0, header=0)
liczba_wierszy, liczba_kolumn = macierz.shape
liczba_miast = macierz.shape[0]
macierz_numpy = macierz.to_numpy()
print(macierz_numpy)