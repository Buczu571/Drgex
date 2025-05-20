import os
import numpy as np
import csv
import tensorflow as tf
from keras.api.models import *
from keras.api.layers import *
from keras.api.utils import *
from scipy.signal import iirnotch, filtfilt
import librosa
from scipy.signal import savgol_filter

SAMPLING_RATE = 24000
SAMPLE_SEC = 12

windows_alumi = []
windows_jodla = []
windows_polia = []
base_dir = os.path.dirname(os.path.abspath(__file__))

csv_files_alumi = [f for f in os.listdir(os.path.join(base_dir, "alumi")) if f.endswith(".csv")]
csv_files_jodla = [f for f in os.listdir(os.path.join(base_dir, "jodla")) if f.endswith(".csv")]
csv_files_polia = [f for f in os.listdir(os.path.join(base_dir, "polia")) if f.endswith(".csv")]

def notch_filter(data, freq, fs, quality):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

for file_name in csv_files_alumi:
    with open(os.path.join(base_dir, "alumi", file_name), mode="r", newline="") as file:
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
                windows_alumi.append(current_window)
                current_window = []
            row_counter = row_counter + 1

for file_name in csv_files_jodla:
    with open(os.path.join(base_dir, "jodla", file_name), mode="r", newline="") as file:
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
                windows_jodla.append(current_window)
                current_window = []
            row_counter = row_counter + 1

for file_name in csv_files_polia:
    with open(os.path.join(base_dir, "polia", file_name), mode="r", newline="") as file:
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
                windows_polia.append(current_window)
                current_window = []
            row_counter = row_counter + 1


fft_alumi = []
fft_jodla = []
fft_polia = []


for window in windows_alumi:
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

    fft_alumi.append(fft_magnitude)

for window in windows_jodla:
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

    fft_jodla.append(fft_magnitude)

for window in windows_polia:
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

    fft_polia.append(fft_magnitude)


X = np.array(fft_alumi + fft_jodla + fft_polia)
y = np.array([0.60] * len(fft_alumi) + [0.08] * len(fft_jodla) + [0.20] * len(fft_polia))
X = X[..., np.newaxis] 

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.shape, X_train.shape)

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=4),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=32, batch_size=32)

model.save('model_material.keras')
