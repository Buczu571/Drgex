import os
import numpy as np
import csv
import tensorflow as tf
from keras.api.models import *
from keras.api.layers import *
from scipy.signal import iirnotch, filtfilt
from collections import defaultdict
import librosa
from scipy.signal import savgol_filter


#******************************************** PARAMETRY ********************************************

SAMPLING_RATE = 24000
SAMPLE_SEC = 12

model = tf.keras.models.load_model('model_material.keras')


def notch_filter(data, freq, fs, quality):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

def gaussian_filter(freq, center_freq, sigma):
    return np.exp(-0.5 * ((freq - center_freq) / sigma) ** 2)

windows_test = []
windows_file_name = []
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_files_test = [f for f in os.listdir(os.path.join(base_dir, "t")) if f.endswith(".csv")]

print(csv_files_test)


for file_name in csv_files_test:
    with open(os.path.join(base_dir, "t", file_name), mode="r", newline="") as file:
        reader = csv.reader(file)

        import_data = [int(row[0]) for i, row in enumerate(reader)]  

        import_data.pop(0)

        current_window = []
        row_counter = 0

        for row in import_data:
            if row_counter > SAMPLE_SEC * SAMPLING_RATE:
                break
            current_window.append(row)
            if len(current_window) >= SAMPLING_RATE:
                windows_test.append(current_window)
                windows_file_name.append(file_name)
                current_window = []
            row_counter = row_counter + 1


fft_test = []

for window in windows_test:
    window = notch_filter(window, 50, 24000, 72)

    for i in range(50, 12050, 50):
        window = notch_filter(window, i, 24000, 48)

    window = window - np.mean(window)

    fft_result = np.fft.fft(window)

    fft_freq = np.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

    fft_magnitude = np.abs(fft_result) / (SAMPLING_RATE)

    half_n = SAMPLING_RATE // 2
    fft_freq = fft_freq[:half_n]
    fft_magnitude = fft_magnitude[:half_n]

    fft_freq = fft_freq[:4000]
    fft_magnitude = fft_magnitude[:4000]

    fft_test.append(fft_magnitude)


X_test = np.array(fft_test)
X_test = X_test[..., np.newaxis]
predictions = model.predict(X_test)
predictions = np.array(predictions).flatten()

print(predictions)

aggregated_results  = defaultdict(list)

for result, filename in zip(predictions, windows_file_name):
    aggregated_results [filename].append(result)

averaged_results = {filename: sum(vals)/len(vals) for filename, vals in aggregated_results.items()}

print(averaged_results[filename])
print(type(averaged_results[filename]))

for filename in sorted(averaged_results):
    print(f"Plik \"{filename}\" - Przewidywana twardość materiału: {averaged_results[filename]:.4f} GPa")



