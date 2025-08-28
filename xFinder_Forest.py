import os
import numpy as np
from xFinder_v2 import VibrationAnalyzer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

analyzer = VibrationAnalyzer()
estimated_rpm = 1220

data = []
labels = []

class_folders = {
    0: 'data/good',
    1: 'data/unbalance',
    2: 'data/severe'
}

for label, folder in class_folders.items():
    for fname in os.listdir(folder):
        if not fname.endswith('.csv'):
            continue
        filepath = os.path.join(folder, fname)
        signal = analyzer.load_signal(filepath)
        xf, vspec = analyzer.compute_velocity_spectrum(signal)
        peak_freq, peak_val = analyzer.find_1x_rpm(xf, vspec, estimated_rpm)
        if peak_val is not None:
            data.append([peak_val])
            labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['good', 'unbalance', 'severe']))

joblib.dump(clf, 'unbalance_model.pkl')

