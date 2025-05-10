import os
import numpy as np
import csv
import tensorflow as tf
from keras.api.models import *
from keras.api.layers import *
from scipy.signal import iirnotch, filtfilt

SAMPLING_RATE = 24000
SAMPLE_SEC = 20

windows_bad = []
windows_good = []
base_dir = os.path.dirname(os.path.abspath(__file__))

csv_files_good = [f for f in os.listdir(os.path.join(base_dir, "g")) if f.endswith(".csv")]
csv_files_bad = [f for f in os.listdir(os.path.join(base_dir, "b")) if f.endswith(".csv")]

def notch_filter(data, freq, fs, quality):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

for file_name in csv_files_bad:
    with open(os.path.join(base_dir, "b", file_name), mode="r", newline="") as file:
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
                windows_bad.append(current_window)
                current_window = []
            row_counter = row_counter + 1


for file_name in csv_files_good:
    with open(os.path.join(base_dir, "g", file_name), mode="r", newline="") as file:
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
                windows_good.append(current_window)
                current_window = []
            row_counter = row_counter + 1


fft_bad = []
fft_good = []
fft_test = []


for window in windows_bad:
    for i in range(50, 1000, 50):
        window = notch_filter(window, i, 24000, 12)

    window = window - np.mean(window)

    fft_result = np.fft.fft(window)
    
    fft_freq = np.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

    fft_magnitude = np.abs(fft_result) / (SAMPLING_RATE)
    
    half_n = SAMPLING_RATE // 2
    fft_freq = fft_freq[:half_n]
    fft_magnitude = fft_magnitude[:half_n]
    fft_magnitude = fft_magnitude[:1000]
    

    fft_bad.append(fft_magnitude)

for window in windows_good:
    for i in range(50, 1000, 50):
        window = notch_filter(window, i, 24000, 12)
        
    window = window - np.mean(window)

    fft_result = np.fft.fft(window)
    
    fft_freq = np.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

    fft_magnitude = np.abs(fft_result) / (SAMPLING_RATE)
    
    half_n = SAMPLING_RATE // 2
    fft_freq = fft_freq[:half_n]
    fft_magnitude = fft_magnitude[:half_n]
    fft_magnitude = fft_magnitude[:1000]

    fft_good.append(fft_magnitude)


X = np.array(fft_bad + fft_good)
y = np.array([1] * len(fft_bad) + [0] * len(fft_good))
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
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=32, batch_size=32)

model.save('model_fft_cnn.h5')
