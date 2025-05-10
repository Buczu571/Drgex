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

def gaussian_filter(freq, center_freq, sigma):
    return np.exp(-0.5 * ((freq - center_freq) / sigma) ** 2)

expected_rpm = 1160 #EXPECTED RPM EXCPETED RPM EXPECTED RPM EXPECTED RPM EXPECTED RPM

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
    for i in range(50, 4000, 50):
        window = notch_filter(window, i, 24000, 24)

    window = window - np.mean(window)

    fft_result = np.fft.fft(window)
    
    fft_freq = np.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

    fft_magnitude = np.abs(fft_result) / (SAMPLING_RATE)

    expected_freq = expected_rpm / 60

    window_width_hz = 16
    sigma = window_width_hz / 6

    gaussian_window = np.exp(-0.5 * ((fft_freq - expected_freq) / sigma) ** 2)

    half_n = 4000

    positive_freq = fft_freq[:half_n]
    positive_fft = fft_magnitude[:half_n]
    gaussian_window_positive = gaussian_window[:half_n]

    weighted_fft = np.abs(positive_fft) * gaussian_window_positive

    peak_idx = np.argmax(weighted_fft)
    found_freq = positive_freq[peak_idx]
    
    peak_amplitude = weighted_fft[peak_idx]
    threshold = 0.8 * peak_amplitude

    left_idx = peak_idx - 1 if peak_idx - 1 >= 0 else None
    right_idx = peak_idx + 1 if peak_idx + 1 < len(weighted_fft) else None

    left_freq = positive_freq[left_idx] if left_idx is not None else None
    right_freq = positive_freq[right_idx] if right_idx is not None else None

    left_amplitude = weighted_fft[left_idx] if left_idx is not None else None
    right_amplitude = weighted_fft[right_idx] if right_idx is not None else None

    if left_amplitude is not None and left_amplitude >= threshold:
        found_freq = (found_freq + left_freq) / 2

    if right_amplitude is not None and right_amplitude >= threshold:
        found_freq = (found_freq + right_freq) / 2

    found_freq = round(found_freq * 2) / 2
    print(found_freq)

    f1 = found_freq
    f2 = 2 * f1
    f3 = 3 * f1
    f4 = 3.5 * f1

    positive_freqs = fft_freq[:half_n]
    positive_power = fft_magnitude[:half_n]

    #max_freq = f4
    #mask = positive_freqs <= max_freq
    #limited_freqs = positive_freqs[mask]
    #limited_power = positive_power[mask]

    window_width_hz = 4
    sigma = window_width_hz / 6

    gaussian_f1 = gaussian_filter(np.array(positive_freqs), f1, sigma)
    gaussian_f2 = gaussian_filter(np.array(positive_freqs), f2, sigma)
    gaussian_f3 = gaussian_filter(np.array(positive_freqs), f3, sigma)

    filtered_power_f1 = np.array(positive_power) * gaussian_f1
    filtered_power_f2 = np.array(positive_power) * gaussian_f2
    filtered_power_f3 = np.array(positive_power) * gaussian_f3

    filtered_power = filtered_power_f1 + filtered_power_f2 + filtered_power_f3
    filtered_freq = np.array(positive_freqs)

    fft_bad.append(filtered_power)
    

for window in windows_good:
    for i in range(50, 4000, 50):
        window = notch_filter(window, i, 24000, 24)

    window = window - np.mean(window)

    fft_result = np.fft.fft(window)
    
    fft_freq = np.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

    fft_magnitude = np.abs(fft_result) / (SAMPLING_RATE)
    
    expected_freq = expected_rpm / 60

    window_width_hz = 16
    sigma = window_width_hz / 6

    gaussian_window = np.exp(-0.5 * ((fft_freq - expected_freq) / sigma) ** 2)

    half_n = 4000

    positive_freq = fft_freq[:half_n]
    positive_fft = fft_magnitude[:half_n]
    gaussian_window_positive = gaussian_window[:half_n]

    weighted_fft = np.abs(positive_fft) * gaussian_window_positive

    peak_idx = np.argmax(weighted_fft)
    found_freq = positive_freq[peak_idx]
    
    peak_amplitude = weighted_fft[peak_idx]
    threshold = 0.8 * peak_amplitude

    left_idx = peak_idx - 1 if peak_idx - 1 >= 0 else None
    right_idx = peak_idx + 1 if peak_idx + 1 < len(weighted_fft) else None

    left_freq = positive_freq[left_idx] if left_idx is not None else None
    right_freq = positive_freq[right_idx] if right_idx is not None else None

    left_amplitude = weighted_fft[left_idx] if left_idx is not None else None
    right_amplitude = weighted_fft[right_idx] if right_idx is not None else None

    if left_amplitude is not None and left_amplitude >= threshold:
        found_freq = (found_freq + left_freq) / 2

    if right_amplitude is not None and right_amplitude >= threshold:
        found_freq = (found_freq + right_freq) / 2

    found_freq = round(found_freq * 2) / 2
    print(found_freq)

    f1 = found_freq
    f2 = 2 * f1
    f3 = 3 * f1
    f4 = 3.5 * f1

    positive_freqs = fft_freq[:half_n]
    positive_power = fft_magnitude[:half_n]

    #max_freq = f4
    #mask = positive_freqs <= max_freq
    #limited_freqs = positive_freqs[mask]
    #limited_power = positive_power[mask]

    window_width_hz = 4
    sigma = window_width_hz / 6

    gaussian_f1 = gaussian_filter(np.array(positive_freqs), f1, sigma)
    gaussian_f2 = gaussian_filter(np.array(positive_freqs), f2, sigma)
    gaussian_f3 = gaussian_filter(np.array(positive_freqs), f3, sigma)

    filtered_power_f1 = np.array(positive_power) * gaussian_f1
    filtered_power_f2 = np.array(positive_power) * gaussian_f2
    filtered_power_f3 = np.array(positive_power) * gaussian_f3

    filtered_power = filtered_power_f1 + filtered_power_f2 + filtered_power_f3
    filtered_freq = np.array(positive_freqs)

    fft_good.append(filtered_power)

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

model.save('model_fft_cnn2.h5')
