import os
import csv
import numpy as np
import tensorflow as tf
from scipy.signal import iirnotch, filtfilt

class TestSignalProcessor:
    def __init__(self, sample_rate=24000, duration=20):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * duration
        self.window_size = sample_rate
        self.fft_limit = 1000

    def load_test_data(self, folder_path):
        """Wczytuje dane testowe i mapuje na pliki."""
        test_windows = []
        file_associations = []
        for file in os.listdir(folder_path):
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Pomijanie nagłówka
                values = [int(row[0]) for row in reader]
                for i in range(0, min(len(values), self.max_samples), self.window_size):
                    window = values[i:i + self.window_size]
                    if len(window) == self.window_size:
                        test_windows.append(window)
                        file_associations.append(file)
        return test_windows, file_associations

    def process_test_signal(self, window):
        """Przetwarzanie sygnału: filtr notch i FFT."""
        signal = np.array(window, dtype=float)
        for freq in range(50, 1000, 50):
            b, a = iirnotch(freq, 12, self.sample_rate)
            signal = filtfilt(b, a, signal)
        signal -= np.mean(signal)
        fft_result = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_result)[:self.sample_rate // 2] / self.sample_rate
        return fft_magnitude[:self.fft_limit]

def main():
    processor = TestSignalProcessor()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Wczytywanie modelu
    model = tf.keras.models.load_model('model_fft_gru.h5')

    # Wczytywanie danych testowych
    test_windows, file_associations = processor.load_test_data(os.path.join(base_dir, 't'))

    # Przetwarzanie FFT
    test_features = [processor.process_test_signal(window) for window in test_windows]
    X_test = np.array(test_features)[..., np.newaxis]

    # Lista unikalnych plików
    unique_files = sorted(list(set(file_associations)))
    print("Pliki testowe:", unique_files)

    # Predykcje
    probabilities = model.predict(X_test).flatten() * 100

    # Agregacja wyników dla plików
    file_probs = {file: [] for file in unique_files}
    for file, prob in zip(file_associations, probabilities):
        file_probs[file].append(prob)

    # Wyświetlenie wyników
    print("\nWyniki klasyfikacji:")
    for file in unique_files:
        avg_prob = np.mean(file_probs[file])
        label = "Uszkodzony" if avg_prob >= 50 else "Zdrowy"
        print(f"Plik: {file}, Średnie prawdopodobieństwo: {avg_prob:.2f}%, Klasyfikacja: {label}")

if __name__ == '__main__':
    main()