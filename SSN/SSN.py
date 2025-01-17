import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input
from keras._tf_keras.keras.optimizers import Adam, SGD, RMSprop
from keras._tf_keras.keras.callbacks import EarlyStopping

# Funkcja obliczająca odległość w km między dwoma punktami
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Promień Ziemi w kilometrach
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Wczytaj dane
wd = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(wd+'/amazon_delivery.csv')

data['Distance'] = haversine(
    data['Store_Latitude'], data['Store_Longitude'],
    data['Drop_Latitude'], data['Drop_Longitude']
)

# Lista kolumn do One-Hot Encoding
one_hot_columns = ['Weather', 'Traffic', 'Vehicle', 'Area']
# One-Hot Encoding dla kolumn z mniejszą liczbą unikalnych wartości dla danych kategorycznych
one_hot_encoded = pd.get_dummies(data[one_hot_columns], drop_first=True)
# Sprawdzenie kolumn po One-Hot Encoding
print("Kolumny po One-Hot Encoding:")
print(one_hot_encoded.columns)
# Scalanie danych
data = pd.concat([data, one_hot_encoded], axis=1)

# Label Encoding dla kolumn z większą liczbą unikalnych wartości
data['Category_Encoded'] = LabelEncoder().fit_transform(data['Category'])

# Konwersja kolumn daty i czasu
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data['Order_Time'] = pd.to_datetime(data['Order_Time'].str.strip(), format='%H:%M:%S', errors='coerce')
data['Pickup_Time'] = pd.to_datetime(data['Pickup_Time'].str.strip(), format='%H:%M:%S', errors='coerce')

data['Order_Weekday'] = data['Order_Date'].dt.weekday
data['Order_Hour'] = data['Order_Time'].dt.hour
data['Pickup_Hour'] = data['Pickup_Time'].dt.hour

# Sprawdzenie wszystkich kolumn w data
print("Wszystkie kolumny w data:")
print(data.columns)

data.columns = data.columns.str.strip()

# Przygotowanie danych
selected_columns = [
    'Agent_Age', 'Agent_Rating', 'Distance', 'Order_Weekday', 'Order_Hour', 'Pickup_Hour',
    'Weather_Fog', 'Weather_Sandstorms', 'Weather_Stormy', 'Weather_Sunny', 'Weather_Windy',
    'Traffic_Jam', 'Traffic_Low', 'Traffic_Medium', 'Vehicle_motorcycle', 'Vehicle_scooter',
    'Vehicle_van', 'Area_Semi-Urban', 'Area_Urban', 'Category_Encoded'
]

# Przygotowanie zmiennych
X = data[selected_columns].fillna(0)  # Zastępowanie brakujących wartości
y = data['Delivery_Time'].astype(float)

print(data.info())  # Informacje o typach danych i brakujących wartościach
print(data.describe())  # Statystyki opisowe dla kolumn numerycznych
data = data.dropna()
print(data.info())  # Informacje o typach danych i brakujących wartościach
print(data.describe())


# Funkcja budująca i trenująca model
def train_model(X, y, 
                num_layers=3, neurons_per_layer=64, 
                activation='relu',
                optimizer='adam', learning_rate=0.001, 
                batch_size=32, epochs=50, 
                test_size=0.2, validation_split=0.1,
                callbacks=None, log_results=True, experiment_name="default"):
    """
    Trenuje model sieci neuronowej z różnymi parametrami.
    
    Argumenty:
    - X: Dane wejściowe
    - y: Etykiety (czas dostawy lub klasy)
    - num_layers: Liczba warstw ukrytych
    - neurons_per_layer: Liczba neuronów na warstwę
    - activation: Funkcja aktywacji dla warstw ukrytych
    - final_activation: Funkcja aktywacji dla ostatniej warstwy
    - optimizer: Optymalizator ('adam', 'sgd', 'rmsprop')
    - learning_rate: Szybkość uczenia
    - batch_size: Rozmiar batcha
    - epochs: Liczba epok
    - test_size: Proporcja danych testowych
    - validation_split: Proporcja danych walidacyjnych
    - callbacks: Lista callbacków (np. EarlyStopping)
    - log_results: Czy zapisywać wyniki w pliku CSV
    - experiment_name: Nazwa eksperymentu (np. "relu_layers3")

    Zwraca:
    - model: Wytrenowany model
    - history: Historia treningu
    - metrics: Wyniki na danych testowych
    """
    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=np.random.randint(1000))

    # Normalizacja danych wejściowych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Budowa modelu
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for _ in range(num_layers - 1): 
        model.add(Dense(neurons_per_layer, activation=activation))
    model.add(Dense(1, activation='linear'))
    
    # Kompilacja modelu
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Nieprawidłowy optymalizator. Wybierz 'adam', 'sgd', lub 'rmsprop'.")
    
    # Kompilacja modelu 
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    
    # Callbacki
    if callbacks is None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        callbacks = [early_stopping]
    
    # Trening modelu
    history = model.fit(X_train, y_train, 
                        validation_split=validation_split, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=callbacks, 
                        verbose=0)
    
    # Ocena modelu
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {'Test Loss': loss, 'Test MAE': mae, 'Test MSE': mse}
    print(f"[{experiment_name}] Test Loss: {loss:.4f}, Test MAE: {mae:.4f}, Test MSE: {mse:.4f}")
    
    # Logowanie wyników
    if log_results:
        results = {
            'Experiment': experiment_name,
            'Layers': num_layers,
            'Neurons per Layer': neurons_per_layer,
            'Activation': activation,
            'Optimizer': optimizer,
            'Learning Rate': learning_rate,
            'Batch Size': batch_size,
            'Epochs': epochs,
            'Validation Split': validation_split,
            'Test Loss': loss,
            'Test MAE': mae,
            'Test MSE': mse
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv("experiment_results.csv", mode='a', header=not pd.io.common.file_exists("experiment_results.csv"), index=False)
        print(f"Experiment '{experiment_name}' completed. Results saved to 'experiment_results.csv'.")

    # Rysowanie wykresów
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Training Curve [{experiment_name}]")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model, history, metrics

model, history, metrics = train_model(
    X, y,
    num_layers=4,               # Więcej warstw
    neurons_per_layer=256,      # Więcej neuronów
    activation='relu',          # Funkcja aktywacji
    optimizer='adam',           # Optymalizator
    learning_rate=0.001,        # Domyślna szybkość uczenia
    batch_size=16,              # Średni batch
    epochs=100,                  # Więcej epok
    experiment_name="improved_model1"
)
