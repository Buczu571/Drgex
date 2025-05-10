import os
import numpy as np
import csv
import tensorflow as tf
from keras.api.models import *
from keras.api.layers import *
from scipy.signal import iirnotch, filtfilt
from collections import defaultdict


#******************************************** PARAMETRY ********************************************

SAMPLING_RATE = 24000
SAMPLE_SEC = 20
expected_rpm = 1160

model = tf.keras.models.load_model('model_fft_cnn.h5')
model2 = tf.keras.models.load_model('model_fft_cnn2.h5')

def notch_filter(data, freq, fs, quality):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

def gaussian_filter(freq, center_freq, sigma):
    return np.exp(-0.5 * ((freq - center_freq) / sigma) ** 2)

windows_test = []
windows_file_name = []
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_files_test = [f for f in os.listdir(os.path.join(base_dir, "t")) if f.endswith(".csv")]


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


fft_test1 = []
fft_test2 = []

for window in windows_test:
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
    
    fft_test1.append(fft_magnitude)


for window in windows_test:
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

    f1 = found_freq
    f2 = 2 * f1
    f3 = 3 * f1

    positive_freqs = fft_freq[:half_n]
    positive_power = fft_magnitude[:half_n]

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
    
    fft_test2.append(filtered_power)


predictions_percent_model1 = []
predictions_percent_model2 = []

X_test1 = np.array(fft_test1)
X_test1 = X_test1[..., np.newaxis]
X_test2 = np.array(fft_test2)
X_test2 = X_test2[..., np.newaxis]

print(csv_files_test)

predictions1 = model.predict(X_test1)
predictions2 = model2.predict(X_test2)

predictions_percent1 = predictions1.flatten() * 100  
predictions_percent2 = predictions2.flatten() * 100

for i, prob in enumerate(predictions_percent1):
    predictions_percent_model1.append(prob)

for i, prob in enumerate(predictions_percent2):
    predictions_percent_model2.append(prob)

aggregated_results = defaultdict(list)

for filename, result1, result2 in zip(windows_file_name, predictions_percent_model1, predictions_percent_model2):
    aggregated_results[filename].append(result1)
    aggregated_results[filename].append(result2)

averaged_results = {filename: sum(vals)/len(vals) for filename, vals in aggregated_results.items()}

for filename in sorted(averaged_results):
    print(f"Prawdopodobieństwo niewyważenia, próbka \"{filename}\": {averaged_results[filename]:.2f}%")




