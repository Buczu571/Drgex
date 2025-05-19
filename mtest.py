import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

class VibrationClassifier:
    def __init__(self, model_path):
        # Wczytaj zapisany model
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler()
        self.scaler.scale_ = np.load(f'{model_path}_scaler.npy')
        self.scaler.min_ = np.load(f'{model_path}_min.npy')
        self.class_names = np.load(f'{model_path}_classes.npy', allow_pickle=True)
        self.sequence_length = self.model.input_shape[1]  # Automatyczne wykrycie długości sekwencji

    def predict_from_csv(self, file_path):
        """Wczytaj pojedynczy plik CSV i wykonaj predykcję"""
        # Wczytaj dane
        data = pd.read_csv(file_path)
        
        # Automatyczne wykrycie kolumny z danymi
        if 'vibration' in data.columns:
            vib_col = 'vibration'
        elif len(data.columns) == 1:
            vib_col = data.columns[0]
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            vib_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if not vib_col:
            raise ValueError("Nie znaleziono kolumny z danymi w pliku CSV")
        
        print(f"Używam kolumny: '{vib_col}'")
        raw_data = data[[vib_col]].values
        
        # Przetwarzanie danych
        scaled_data = self.scaler.transform(raw_data)
        
        # Podziel na sekwencje
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            sequences.append(scaled_data[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        # Predykcja
        predictions = self.model.predict(sequences, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Interpretacja wyników
        results = [self.class_names[cls] for cls in predicted_classes]
        confidence = np.max(predictions, axis=1)
        
        return results, confidence

if __name__ == "__main__":
    # Ścieżka do wytrenowanego modelu
    MODEL_PATH = 'best_model.h5'
    
    # Inicjalizacja klasyfikatora
    classifier = VibrationClassifier(MODEL_PATH)
    
    # Przykładowy plik do analizy
    TEST_FILE = 'nowe_dane.csv'
    
    # Wykonaj predykcję
    try:
        results, confidence = classifier.predict_from_csv(TEST_FILE)
        
        # Wyświetl wyniki
        print("\nWyniki predykcji:")
        for i, (result, conf) in enumerate(zip(results, confidence)):
            print(f"Sekwencja {i+1}: {result} (pewność: {conf:.2%})")
    except Exception as e:
        print(f"\nBłąd podczas przetwarzania: {str(e)}")