import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Helper Functions
def load_data(file_path, skip_first_line=False, skip_samples=0, fs=1000, min_duration=1.0, target_length=72000):
    """Load data from a CSV file and normalize to fixed length."""
    try:
        data = pd.read_csv(file_path, header=None).values.flatten()
        if skip_first_line:
            data = data[1:]
        if len(data) <= skip_samples:
            print(f"File {file_path} too short after skipping samples - skipping")
            return None
        data = data[skip_samples:]
        min_samples = int(fs * min_duration)
        if len(data) < min_samples:
            print(f"File {file_path} too short ({len(data)} samples < {min_samples}) - skipping")
            return None
        # Normalize to 72000 samples
        if len(data) > target_length:
            data = data[:target_length]
        elif len(data) < target_length:
            data = np.pad(data, (0, target_length - len(data)), mode='constant')
        print(f"Loaded file {file_path} with {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_files_from_folder(folder_path, skip_first_line=True, fs=1000, min_duration=1.0):
    """Load all CSV files from a folder."""
    all_data = []
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist - skipping")
        return all_data
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            data = load_data(file_path, skip_first_line=skip_first_line, fs=fs, min_duration=min_duration)
            if data is not None:
                all_data.append(data)
                print(f"  Loaded: {file} (samples: {len(data)}, duration: {len(data)/fs:.2f} s)")
    return all_data

def extract_features(signal, fs=1000):
    """Extract time and frequency domain features from the entire signal."""
    try:
        # Time-domain features
        rms = np.sqrt(np.mean(signal**2))
        peak_to_peak = np.max(signal) - np.min(signal)
        mean_abs = np.mean(np.abs(signal))
        variance = np.var(signal)
        skewness = np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3 + 1e-10)
        kurtosis = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-10)
        crest_factor = peak_to_peak / (rms + 1e-10)  # Added crest factor

        # Frequency-domain features
        N = len(signal)
        fft_vals = np.abs(fft(signal))[:N//2]
        fft_vals = fft_vals / (np.sum(fft_vals) + 1e-10)
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        dominant_freq_idx = np.argmax(fft_vals)
        dominant_freq = freqs[dominant_freq_idx]
        spectral_entropy = -np.sum(fft_vals * np.log(fft_vals + 1e-10))
        low_freq_band = np.sum(fft_vals[(freqs >= 0) & (freqs < 50)])
        mid_freq_band = np.sum(fft_vals[(freqs >= 50) & (freqs < 200)])
        high_freq_band = np.sum(fft_vals[(freqs >= 200) & (freqs < fs/2)])

        return np.array([
            rms, peak_to_peak, mean_abs, variance, skewness, kurtosis, crest_factor,
            dominant_freq, spectral_entropy, low_freq_band, mid_freq_band, high_freq_band
        ])
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return np.zeros(12)  # Adjusted for 12 features

def augment_signal(signal, noise_factor=0.1):
    """Augment signal by adding noise, scaling, and time shifting."""
    augmented = signal.copy()
    # Add noise
    noise = np.random.normal(0, noise_factor * np.std(augmented), augmented.shape)
    augmented = augmented + noise
    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented = augmented * scale
    # Time shift
    shift = np.random.randint(-1000, 1000)
    augmented = np.roll(augmented, shift)
    return augmented

def visualize_fft(signal, fs=1000, save_path=None):
    """Visualize FFT spectrum of the signal."""
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N//2] / (np.sum(np.abs(fft(signal))[:N//2]) + 1e-10)
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_vals)
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Parametry
random_state = 42
project_path = r"C:\Users\Justyna\OneDrive\Pulpit\RNN"
fs = 1000
min_duration = 1.0

healthy_folders = ['praca_normalna', 'praca_normalna2']
faulty_folders = ['niewywazenie_sruba', 'niewywazeniezmetalowymobciaznikiem', 'wiertło_15_04']
healthy_files = ['zdrowat1.csv', 'zdrowat2.csv', 'zdrowat3.csv', 'zdrowat4.csv', 'zdrowat5.csv']
faulty_files = ['zle1.csv', 'zle2.csv', 'zle3.csv']

# Load Healthy Data
print("Loading healthy data...")
healthy_data = []
for file in healthy_files:
    file_path = os.path.join(project_path, file)
    if os.path.exists(file_path):
        data = load_data(file_path, skip_first_line=True, fs=fs, min_duration=min_duration)
        if data is not None:
            healthy_data.append(data)
            print(f"Loaded file: {file} (samples: {len(data)})")
    else:
        print(f"File {file_path} does not exist - skipping")

for folder in healthy_folders:
    folder_path = os.path.join(project_path, folder)
    print(f"\nProcessing healthy folder: {folder}")
    healthy_data.extend(load_all_files_from_folder(folder_path, skip_first_line=True, fs=fs, min_duration=min_duration))

if not healthy_data:
    print("No healthy data loaded - exiting")
    exit(1)

# Load Faulty Data
print("\nLoading faulty data...")
faulty_data = []
for file in faulty_files:
    file_path = os.path.join(project_path, file)
    if os.path.exists(file_path):
        data = load_data(file_path, skip_first_line=True, fs=fs, min_duration=min_duration)
        if data is not None:
            faulty_data.append(data)
            print(f"Loaded file: {file} (samples: {len(data)})")
    else:
        print(f"File {file_path} does not exist - skipping")

for folder in faulty_folders:
    folder_path = os.path.join(project_path, folder)
    print(f"\nProcessing faulty folder: {folder}")
    faulty_data.extend(load_all_files_from_folder(folder_path, skip_first_line=True, fs=fs, min_duration=min_duration))

if not faulty_data:
    print("No faulty data loaded - exiting")
    exit(1)

# Visualize FFT
if healthy_data:
    visualize_fft(healthy_data[0], fs=fs, save_path=os.path.join(project_path, "fft_healthy_rnn_gru.png"))
if faulty_data:
    visualize_fft(faulty_data[0], fs=fs, save_path=os.path.join(project_path, "fft_faulty_rnn_gru.png"))

# Feature Extraction and Augmentation
X_healthy = [extract_features(signal, fs=fs) for signal in healthy_data]
y_healthy = [0] * len(X_healthy)

X_faulty = [extract_features(signal, fs=fs) for signal in faulty_data]
y_faulty = [1] * len(X_faulty)

# Data Augmentation (10 for healthy, 20 for faulty)
X_healthy_augmented = []
y_healthy_augmented = []
for signal in healthy_data:
    for noise_factor in [0.2, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005]:
        aug_signal = augment_signal(signal, noise_factor=noise_factor)
        X_healthy_augmented.append(extract_features(aug_signal, fs=fs))
        y_healthy_augmented.append(0)

X_faulty_augmented = []
y_faulty_augmented = []
for signal in faulty_data:
    for noise_factor in [0.2, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005]:
        aug_signal = augment_signal(signal, noise_factor=noise_factor)
        X_faulty_augmented.append(extract_features(aug_signal, fs=fs))
        y_faulty_augmented.append(1)

# Combine Data
X_healthy = X_healthy + X_healthy_augmented
y_healthy = y_healthy + y_healthy_augmented
X_faulty = X_faulty + X_faulty_augmented
y_faulty = y_faulty + y_faulty_augmented

X = X_healthy + X_faulty
y = y_healthy + y_faulty

print(f"\nNumber of samples after augmentation:")
print(f"Healthy: {len(X_healthy)}")
print(f"Faulty: {len(X_faulty)}")
if len(X_healthy) == 0 or len(X_faulty) == 0:
    print("Insufficient samples - exiting")
    exit(1)

# Normalize Features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X = X_normalized

# Reshape for GRU (samples, timesteps=1, features=12)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Convert Labels to Array
y = np.array(y)

# Split Data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_state
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=random_state
)

# Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))
class_weights[0] *= 1.2  # Slightly increase weight for healthy class
class_weights[1] *= 1.5  # Increase weight for faulty class to prioritize recall

# RNN-GRU Model
model = Sequential([
    GRU(128, input_shape=(1, 12), kernel_regularizer=l2(0.001), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    GRU(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR')
    ]
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        mode='max',
        restore_best_weights=True,
        min_delta=0.01,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=15,
        min_lr=1e-9,
        verbose=1,
        mode='max'
    )
]

# Train Model
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluate Model
print("\nEvaluating model on test set:")
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f"Accuracy: {test_results[1]:.4f}")
print(f"Precision: {test_results[2]:.4f}")
print(f"Recall: {test_results[3]:.4f}")
print(f"AUC: {test_results[4]:.4f}")
print(f"PR-AUC: {test_results[5]:.4f}")

# Evaluate with Different Thresholds
y_pred = model.predict(X_test)
for threshold in [0.5, 0.3, 0.2]:
    y_pred_binary = (y_pred > threshold).astype(int)
    print(f"\nEvaluation with threshold {threshold}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_binary):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_binary):.4f}")

# ROC and PR Curves
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(project_path, 'roc_pr_curves_rnn_gru.png'))
plt.close()

# Training Plots
plt.figure(figsize=(15, 10))
for i, (metric, title) in enumerate([
    ('accuracy', 'Model Accuracy'),
    ('loss', 'Loss Function'),
    ('recall', 'Recall'),
    ('pr_auc', 'PR-AUC')
], 1):
    plt.subplot(2, 2, i)
    plt.plot(history.history[metric], label='Training')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(project_path, 'training_plots_rnn_gru.png'))
print("\nPlots saved as 'training_plots_rnn_gru.png' and 'roc_pr_curves_rnn_gru.png'")

# Save Model
model.save(os.path.join(project_path, 'model_rnn_gru_vibration.keras'))
print("\nModel saved as 'model_rnn_gru_vibration.keras'")