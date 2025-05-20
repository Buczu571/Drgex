import os
import csv
import numpy as np
from scipy.signal import iirnotch, filtfilt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split

class SignalProcessor:
    def __init__(self, sample_rate=24000, duration=20):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * duration
        self.window_size = sample_rate
        self.fft_limit = 1000

    def read_csv_data(self, folder_path):
        """Wczytuje dane z folderu i dzieli na okna."""
        windows = []
        for file in os.listdir(folder_path):
            if not file.endswith('.csv'):
                continue
            with open(os.path.join(folder_path, file), 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Pomijanie nagłówka
                values = [int(row[0]) for row in reader]
                for i in range(0, min(len(values), self.max_samples), self.window_size):
                    window = values[i:i + self.window_size]
                    if len(window) == self.window_size:
                        windows.append(window)
        return windows

    def process_signal(self, window):
        """Przetwarzanie sygnału: filtr notch i FFT."""
        signal = np.array(window, dtype=float)
        for freq in range(50, 1000, 50):
            b, a = iirnotch(freq, 12, self.sample_rate)
            signal = filtfilt(b, a, signal)
        signal -= np.mean(signal)
        fft_result = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_result)[:self.sample_rate // 2] / self.sample_rate
        return fft_magnitude[:self.fft_limit]

def build_gru_model(input_shape):
    """Tworzenie modelu GRU."""
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        GRU(32),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    processor = SignalProcessor()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Wczytywanie danych
    bad_data = processor.read_csv_data(os.path.join(base_dir, 'b'))
    good_data = processor.read_csv_data(os.path.join(base_dir, 'g'))

    # Przetwarzanie FFT
    bad_features = [processor.process_signal(window) for window in bad_data]
    good_features = [processor.process_signal(window) for window in good_data]

    # Przygotowanie danych
    X = np.array(bad_features + good_features)
    y = np.concatenate([np.ones(len(bad_features)), np.zeros(len(good_features))])
    X = X[..., np.newaxis]

    # Podział danych
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Całkowity kształt danych: {X.shape}, Kształt treningowy: {X_train.shape}")

    # Budowa i trening modelu
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=32, batch_size=32)

    # Zapis modelu
    model.save('model_fft_gru.h5')

if __name__ == '__main__':
    main()