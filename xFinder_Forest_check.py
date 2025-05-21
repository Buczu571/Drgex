import joblib
import numpy as np
from xFinder_v2 import VibrationAnalyzer 

clf = joblib.load('unbalance_model.pkl')
new_sample_path = 'sruba.csv' 


analyzer = VibrationAnalyzer()

estimated_rpm = 1220
signal = analyzer.load_signal(new_sample_path)
xf, vspec = analyzer.compute_velocity_spectrum(signal)
peak_freq, peak_val = analyzer.find_1x_rpm(xf, vspec, estimated_rpm)

if peak_val is None:
    print("Nie znaleziono wartości piku w zakresie 1x RPM.")
else:
    print(f"Obliczona wartość piku: {peak_val:.2f} mm/s")
    predicted_class = clf.predict([[peak_val]])[0]
    class_names = ['good', 'unbalance', 'severe']
    print(f"Przewidywana klasa: {class_names[predicted_class]}")
