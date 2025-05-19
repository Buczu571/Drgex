import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

class VibrationClassifier:
    def __init__(self, sequence_length=100):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.class_names = ['Stan dobry', 'Niewyważenie', 'Poluzowane imadło', 'Za głębokie wiercenie']
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_dir = os.path.join(self.script_dir, 'data', 'train')
        self.test_dir = os.path.join(self.script_dir, 'data', 'test')
        self.models_dir = os.path.join(self.script_dir, 'models')
        self.model_name = "trained_model"
        
        # Utwórz folder models jeśli nie istnieje
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Utwórz podfoldery dla każdej klasy jeśli nie istnieją
        for state in self.class_names:
            os.makedirs(os.path.join(self.train_dir, state), exist_ok=True)
            os.makedirs(os.path.join(self.test_dir, state), exist_ok=True)
    
    def load_data(self, directory):
        """Ładuje dane z podanego katalogu"""
        features = []
        labels = []
        
        for label, state in enumerate(self.class_names):
            state_dir = os.path.join(directory, state)
            if not os.path.exists(state_dir):
                print(f"Uwaga: Brak folderu {state_dir}")
                continue
                
            files = [f for f in os.listdir(state_dir) if f.endswith('.csv')]
            if not files:
                print(f"Uwaga: Brak plików CSV w {state_dir}")
                continue

            for file in files:
                file_path = os.path.join(state_dir, file)
                try:
                    data = pd.read_csv(file_path)
                    
                    # Automatyczne wykrywanie kolumny z danymi
                    if 'vibration' in data.columns:
                        vib_col = 'vibration'
                    elif len(data.columns) == 1:
                        vib_col = data.columns[0]
                    else:
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        vib_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                    
                    if not vib_col:
                        print(f"Błąd: Nie znaleziono kolumny z danymi w {file}")
                        continue
                    
                    vib_data = data[[vib_col]].values
                    features.append(vib_data)
                    labels.extend([label] * (len(vib_data) - self.sequence_length + 1))
                    
                except Exception as e:
                    print(f"Błąd przetwarzania {file}: {str(e)}")
                    continue
    
        if not features:
            raise ValueError("Nie znaleziono żadnych poprawnych danych wejściowych")
        
        return features, np.array(labels)
    
    def create_sequences(self, data):
        """Tworzy sekwencje z danych"""
        sequences = []
        for sample in data:
            for i in range(len(sample) - self.sequence_length + 1):
                sequences.append(sample[i:i + self.sequence_length])
        return np.array(sequences)
    
    def preprocess_data(self, features, labels):
        """Przygotowuje dane do treningu"""
        flattened = np.concatenate(features).reshape(-1, 1)
        self.scaler.fit(flattened)
        
        sequences = self.create_sequences(features)
        sequences_scaled = np.array([self.scaler.transform(seq) for seq in sequences])
        
        y = to_categorical(labels, num_classes=len(self.class_names))
        
        return sequences_scaled, y
    
    def build_model(self, input_shape):
        """Buduje model LSTM"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, epochs=10, batch_size=64, validation_split=0.2):
        """Trenuje model"""
        print("\n=== Rozpoczęcie treningu ===")
        print(f"Wczytywanie danych z: {self.train_dir}")
        
        try:
            features, labels = self.load_data(self.train_dir)
            X, y = self.preprocess_data(features, labels)
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
            
            input_shape = (self.sequence_length, 1)
            self.model = self.build_model(input_shape)
            
            callbacks = [
                EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                ModelCheckpoint(os.path.join(self.models_dir, 'best_model.h5'), 
                              save_best_only=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.1, patience=3)
            ]
            
            print("\nRozpoczynam trening...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.plot_training_history(history)
            model_name = input("\nPodaj nazwę dla nowego modelu (bez rozszerzenia .h5): ")
            self.save_model(model_name)
            
            return history
        
        except Exception as e:
            print(f"\nBłąd podczas treningu: {str(e)}")
            print("Upewnij się, że:")
            print("1. Folder 'data/train' zawiera podfoldery dla każdej klasy")
            print("2. Każdy podfolder zawiera pliki CSV z danymi")
            return None
    
    def evaluate(self):
        """Ocenia model na danych testowych"""
        if not self.model:
            print("Model niezaładowany — próba automatycznego wczytania...")
            try:
                self.load_model()
            except Exception as e:
                print(f"Błąd wczytywania modelu: {str(e)}")
                return

        print("\n=== Ocena modelu ===")
        print(f"Wczytywanie danych testowych z: {self.test_dir}")
        
        try:
            features, labels = self.load_data(self.test_dir)
            X, y = self.preprocess_data(features, labels)
            
            results = self.model.evaluate(X, y, verbose=0)
            print(f"\nWyniki ewaluacji:")
            print(f"Loss: {results[0]:.4f} | Dokładność: {results[1]:.4f} | Precyzja: {results[2]:.4f} | Czułość: {results[3]:.4f}")
            
            y_pred = self.model.predict(X, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y, axis=1)
            
            print("\nRaport klasyfikacji:")
            print(classification_report(y_true_classes, y_pred_classes, target_names=self.class_names))
            
            self.plot_confusion_matrix(y_true_classes, y_pred_classes)

        except Exception as e:
            print(f"\nBłąd podczas ewaluacji: {str(e)}")
            print("Upewnij się, że folder 'data/test' zawiera poprawne dane testowe")

    
    def predict(self, data_file=None):
        """Wykonuje predykcję na nowych danych"""
        if not self.model:
            print("\nNajpierw wczytaj lub wytrenuj model!")
            return
            
        if data_file is None:
            data_file = input("\nPodaj ścieżkę do pliku CSV do analizy: ")
        
        print(f"\n=== Analiza pliku {data_file} ===")
        
        try:
            data = pd.read_csv(data_file)
            
            # Automatyczne wykrywanie kolumny z danymi
            if 'vibration' in data.columns:
                vib_col = 'vibration'
            elif len(data.columns) == 1:
                vib_col = data.columns[0]
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                vib_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if not vib_col:
                raise ValueError("Nie znaleziono kolumny z danymi")
            
            raw_data = data[[vib_col]].values
            scaled_data = self.scaler.transform(raw_data)
            
            # Przygotuj sekwencje
            sequences = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                sequences.append(scaled_data[i:i + self.sequence_length])
            sequences = np.array(sequences)
            
            # Predykcja
            predictions = self.model.predict(sequences, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            
            # Wyniki
            results = [self.class_names[cls] for cls in predicted_classes]
            
            print("\nWyniki predykcji:")
            for i, (res, conf) in enumerate(zip(results, confidence)):
                print(f"Sekwencja {i+1}: {res} (pewność: {conf:.2%})")
            
            return results, confidence
        
        except Exception as e:
            print(f"\nBłąd podczas predykcji: {str(e)}")
            return None, None
    
    def get_available_models(self):
        """Zwraca listę dostępnych modeli w folderze models"""
        models = []
        if not os.path.exists(self.models_dir):
            print(f"Folder 'models' nie istnieje w: {self.models_dir}")
            return models
            
        for file in os.listdir(self.models_dir):
            if file.endswith('.h5'):
                model_name = file.replace('.h5', '')
                # Sprawdź czy istnieją wszystkie wymagane pliki
                required_files = [
                    f'{model_name}.h5',
                    f'{model_name}.h5_scale.npy',
                    f'{model_name}.h5_min.npy',
                    f'{model_name}.h5_classes.npy'
                ]
                # Poprawione sprawdzanie istnienia plików
                if all(os.path.exists(os.path.join(self.models_dir, f)) for f in required_files):
                    models.append(model_name)
                else:
                    print(f"Brakujące pliki dla modelu: {model_name}")
        return models
    
    def load_model(self):
        """Wczytuje model z folderu models"""
        model_path = os.path.join(self.models_dir, self.model_name)
        
        required_files = [
            f'{self.model_name}.h5',
            f'{self.model_name}.h5_scale.npy',
            f'{self.model_name}.h5_min.npy',
            f'{self.model_name}.h5_classes.npy'
        ]
        
        # Sprawdź czy wszystkie pliki istnieją
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.models_dir, f))]
        if missing_files:
            raise FileNotFoundError(f"Brakujące pliki modelu: {missing_files}")
        
        try:
            # Wczytaj model i powiązane pliki
            print(f"\nWczytywanie modelu {self.model_name}...")
            self.model = load_model(f'{model_path}.h5')
            
            # Ponowna kompilacja modelu
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')]
            )
            
            self.scaler.scale_ = np.load(f'{model_path}.h5_scale.npy')
            self.scaler.min_ = np.load(f'{model_path}.h5_min.npy')
            self.class_names = np.load(f'{model_path}.h5_classes.npy', allow_pickle=True)
            
            print("Pomyślnie wczytano model:")
            print(f"- {model_path}.h5")
            print(f"- {model_path}.h5_scale.npy")
            print(f"- {model_path}.h5_min.npy")
            print(f"- {model_path}.h5_classes.npy")
            return self
        except Exception as e:
            raise RuntimeError(f"Błąd podczas wczytywania modelu: {str(e)}")

    
    def save_model(self):
        """Zapisuje model do folderu models"""
        if not self.model:
            print("\nBrak modelu do zapisania!")
            return
            
        model_path = os.path.join(self.models_dir, self.model_name)
        
        try:
            # Zapis modelu i plików pomocniczych
            self.model.save(f'{model_path}.h5')
            np.save(f'{model_path}.h5_scale.npy', self.scaler.scale_)
            np.save(f'{model_path}.h5_min.npy', self.scaler.min_)
            np.save(f'{model_path}.h5_classes.npy', np.array(self.class_names))
            
            print(f"\nZapisano model '{self.model_name}' w folderze 'models'")
        except Exception as e:
            print(f"\nBłąd podczas zapisywania modelu: {str(e)}")
    
    def plot_training_history(self, history):
        """Wizualizuje proces uczenia"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Trening')
        plt.plot(history.history['val_loss'], label='Walidacja')
        plt.title('Funkcja straty')
        plt.xlabel('Epoka')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Trening')
        plt.plot(history.history['val_accuracy'], label='Walidacja')
        plt.title('Dokładność')
        plt.xlabel('Epoka')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Wyświetla macierz pomyłek"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Macierz pomyłek')
        plt.xlabel('Predykcja')
        plt.ylabel('Rzeczywistość')
        plt.show()


def main():
    print("\n=== Klasyfikator stanów maszyny ===")
    print("1. Trenuj nowy model")
    print("2. Wczytaj istniejący model")
    print("3. Testuj model")
    print("4. Analizuj pojedynczy plik")
    print("0. Wyjdź")
    
    try:
        choice = input("\nWybierz opcję: ")
        
        classifier = VibrationClassifier(sequence_length=100)
        
        if choice == '1':
            epochs = input("Podaj liczbę epok (domyślnie 10): ")
            epochs = int(epochs) if epochs.isdigit() else 10
            classifier.train(epochs=epochs)
        elif choice == '2':
            try:
                classifier.load_model()
                print("\nModel został pomyślnie wczytany!")
            except Exception as e:
                print(f"\nBłąd: {str(e)}")
        elif choice == '3':
            classifier.evaluate()
        elif choice == '4':
            classifier.predict()
        elif choice == '0':
            print("\nZakończono program")
            return
        else:
            print("\nNieprawidłowy wybór")
        
        # Zapytaj czy kontynuować
        if input("\nCzy chcesz kontynuować? (t/n): ").lower() == 't':
            main()
        else:
            print("\nZakończono program")
    
    except Exception as e:
        print(f"\nWystąpił błąd: {str(e)}")
        if input("\nCzy chcesz spróbować ponownie? (t/n): ").lower() == 't':
            main()


if __name__ == "__main__":
    main()