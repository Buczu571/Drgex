import os 
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QTabWidget, QGridLayout, QComboBox, QMessageBox
from PyQt6.QtCore import QTimer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from functools import partial  # Dodaj import partial

class AddMachineWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Referencja do głównego okna
        self.setWindowTitle("Dodaj Maszynę")
        self.setGeometry(100, 100, 300, 200)

        # Layout formularza dodawania maszyny
        self.layout = QVBoxLayout(self)

        # Wprowadzenie nazwy maszyny
        self.machine_name_input = QLineEdit(self)
        self.machine_name_input.setPlaceholderText("Wpisz nazwę maszyny")

        # Wprowadzenie danych technicznych
        self.technical_data_input = QLineEdit(self)
        self.technical_data_input.setPlaceholderText("Wpisz dane techniczne")

        # Wprowadzenie lokalizacji
        self.location_input = QLineEdit(self)
        self.location_input.setPlaceholderText("Wpisz lokalizację")

        # Wprowadzenie dodatkowych informacji
        self.additional_info_input = QLineEdit(self)
        self.additional_info_input.setPlaceholderText("Wpisz dodatkowe informacje")

        # Przycisk zapisz
        self.save_button = QPushButton("Zapisz", self)
        self.save_button.clicked.connect(self.save_machine)  # Połączenie przycisku z metodą save_machine

        # Dodanie widżetów do layoutu
        self.layout.addWidget(self.machine_name_input)
        self.layout.addWidget(self.technical_data_input)
        self.layout.addWidget(self.location_input)
        self.layout.addWidget(self.additional_info_input)
        self.layout.addWidget(self.save_button)

    def save_machine(self):
        # Funkcja zapisywania danych maszyny
        machine_name = self.machine_name_input.text()
        technical_data = self.technical_data_input.text()
        location = self.location_input.text()
        additional_info = self.additional_info_input.text()

        # Dodanie maszyny do listy w głównym oknie
        self.main_window.add_machine(machine_name, technical_data, location, additional_info)

        # Zapisanie danych do pliku
        self.save_to_file(machine_name, technical_data, location, additional_info)

        # Zamknij formularz po zapisaniu
        self.close()
        
    def save_to_file(self, machine_name, technical_data, location, additional_info):
        # Nazwa folderu, w którym będą zapisywane pliki
        folder_name = "machines"
        
        # Utwórz folder, jeśli nie istnieje
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)  # Tworzy folder, jeśli go nie ma
        
        # Tworzenie pełnej ścieżki do pliku
        filename = os.path.join(folder_name, f"{machine_name}.txt")
        
        # Sprawdzenie, czy plik już istnieje
        if os.path.exists(filename):
            # Wyświetlenie komunikatu ostrzegawczego
            QMessageBox.warning(self, "Błąd", f"Plik o nazwie '{filename}' już istnieje. Wybierz inną nazwę.")
            return  # Zakończ funkcję, nie zapisuj pliku
        
        # Jeśli plik nie istnieje, zapisz dane
        with open(filename, 'w') as file:
            file.write(f"Nazwa maszyny: {machine_name}\n")
            file.write(f"Dane techniczne: {technical_data}\n")
            file.write(f"Lokalizacja: {location}\n")
            file.write(f"Dodatkowe informacje: {additional_info}\n")
        
        # Komunikat o pomyślnym zapisie
        QMessageBox.information(self, "Sukces", f"Dane maszyny '{machine_name}' zostały zapisane.")



# Główne okno aplikacji
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Monitorowanie Drgań Maszyn")
        self.setGeometry(100, 100, 800, 600)

        # Zmienna do przechowywania stanu załadowanych danych
        self.data_loaded = False
        self.df = None  # Zmienna na dane z pliku
        self.measurement_time = 0  # Czas trwania pomiaru
        self.timer = QTimer()  # Timer do mierzenia czasu
        self.timer.timeout.connect(self.update_measurement_time)

        # Lista maszyn
        self.machines = []  # Przechowuje informacje o maszynach
        self.machine_labels = []  # Przechowuje etykiety maszyn
        self.measure_buttons = []  # Przechowuje przyciski pomiaru dla maszyn

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Tworzymy główny układ pionowy
        self.main_layout = QVBoxLayout()

        # Tworzymy układ siatkowy (grid) dla przycisków i etykiet
        self.grid_layout = QGridLayout()

        # Przyciski w pierwszym wierszu
        self.measure_without_machine_button = QPushButton("Pomiar bez wybranej maszyny")
        self.file_samples_button = QPushButton("Plik z próbkami")
 

        # Przyciski w drugim wierszu
        self.add_machine_button = QPushButton("Dodaj nową maszynę")
        self.analyze_samples_button = QPushButton("Analiza próbek")

        # Dodanie przycisków do siatki
        self.grid_layout.addWidget(self.measure_without_machine_button, 0, 0)
        self.grid_layout.addWidget(self.file_samples_button, 0, 4)


        self.grid_layout.addWidget(self.add_machine_button, 1, 0)
        self.grid_layout.addWidget(self.analyze_samples_button, 1, 1)

        # Dodanie siatki do głównego layoutu
        self.main_layout.addLayout(self.grid_layout)

        # Inicjalizacja QTabWidget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        central_widget.setLayout(self.main_layout)

        # Połączenie przycisków z funkcjami
        self.add_machine_button.clicked.connect(self.open_add_machine_window)
        self.measure_without_machine_button.clicked.connect(self.open_measurement_without_machine)
        self.file_samples_button.clicked.connect(self.load_file)  # Po kliknięciu ładowanie pliku
        self.analyze_samples_button.clicked.connect(self.open_analyze_samples)

        # Wczytaj maszyny z folderu machines przy starcie aplikacji
        self.load_machines_from_folder()

    def load_machines_from_folder(self):
        # Ścieżka do folderu machines
        folder_name = "machines"
        
        # Sprawdź, czy folder istnieje
        if not os.path.exists(folder_name):
            print(f"Folder '{folder_name}' nie istnieje.")
            return
        
        # Wczytaj pliki z folderu
        for filename in os.listdir(folder_name):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_name, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    machine_data = {
                        "name": lines[0].strip().split(": ")[1],
                        "technical_data": lines[1].strip().split(": ")[1],
                        "location": lines[2].strip().split(": ")[1],
                        "additional_info": lines[3].strip().split(": ")[1]
                    }
                    self.machines.append(machine_data)
        
        # Aktualizacja interfejsu użytkownika
        self.update_machine_ui()

    def update_machine_ui(self):
        # Czyszczenie starych przycisków i etykiet
        for label in self.machine_labels:
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        for button in self.measure_buttons:
            self.grid_layout.removeWidget(button)
            button.deleteLater()

        self.machine_labels.clear()
        self.measure_buttons.clear()

        # Dodanie nowych przycisków i etykiet dla każdej maszyny
        for i, machine in enumerate(self.machines):
            # Etykieta z nazwą maszyny
            machine_label = QLabel(f"Maszyna {i + 1}: {machine['name']}")
            self.machine_labels.append(machine_label)

            # Przycisk "Pomiar dla maszyny"
            measure_button = QPushButton(f"Pomiar dla maszyny {i + 1}")
            measure_button.clicked.connect(partial(self.open_measurement_for_machine, i + 1))  # Użyj partial
            self.measure_buttons.append(measure_button)

            # Przycisk "Edytuj maszynę"
            edit_button = QPushButton("Edytuj maszynę")
            edit_button.clicked.connect(partial(self.edit_machine, i))  # Użyj partial
            self.measure_buttons.append(edit_button)

            # Przycisk "próbki" (bez funkcjonalności)
            sample_info_button = QPushButton("Próbki")
            sample_info_button.setEnabled(False)  # Przycisk nieaktywny
            self.measure_buttons.append(sample_info_button)

            # Dodanie do siatki
            self.grid_layout.addWidget(machine_label, i + 2, 1)  # Wiersz i+2, kolumna 1
            self.grid_layout.addWidget(measure_button, i + 2, 2)  # Wiersz i+2, kolumna 2
            self.grid_layout.addWidget(edit_button, i + 2, 3)  # Wiersz i+2, kolumna 3
            self.grid_layout.addWidget(sample_info_button, i + 2, 4)  # Wiersz i+2, kolumna 4

    def edit_machine(self, machine_index):
        
# Otwieranie pliku o tej samej nazwie co maszyna
        machine_name = self.machines[machine_index]['name']
        folder_name = "machines"
        file_path = os.path.join(folder_name, f"{machine_name}.txt")

        if os.path.exists(file_path):
            # Otwórz plik w domyślnym edytorze tekstu (np. Notatniku na Windowsie)
            os.startfile(file_path)
        else:
            QMessageBox.warning(self, "Błąd", f"Plik '{file_path}' nie istnieje.")
    def open_add_machine_window(self):
        self.add_machine_window = AddMachineWindow(self)  # Przekazujemy referencję do głównego okna
        self.add_machine_window.show()

    def add_machine(self, name, technical_data, location, additional_info):
        # Dodanie maszyny do listy
        self.machines.append({
            "name": name,
            "technical_data": technical_data,
            "location": location,
            "additional_info": additional_info
        })

        # Aktualizacja interfejsu użytkownika
        self.update_machine_ui()

    def load_file(self):
        # Funkcja ładowania pliku
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik", "", "Pliki CSV (*.csv)")
        if file_path:
            self.df = pd.read_csv(file_path)
            self.data_loaded = True
            print(f"Plik załadowany: {file_path}")
        else:
            self.data_loaded = False
            print("Plik nie został załadowany.")

    def open_measurement_without_machine(self):
        if not self.data_loaded:
            print("Błąd: Nie załadowano pliku!")
            return
        
        # Tworzymy nową zakładkę do pomiaru bez maszyny
        measurement_tab = QWidget()  # To jest nowa zakładka
        measurement_layout = QVBoxLayout()

        # Dodanie etykiety z informacją o pomiarze
        measurement_label = QLabel("Pomiar bez wybranej maszyny")
        measurement_layout.addWidget(measurement_label)

        # Dodanie etykiety z czasem trwania pomiaru
        self.duration_label = QLabel("Czas trwania pomiaru: 0s")
        measurement_layout.addWidget(self.duration_label)

        # Przycisk do rozpoczęcia pomiaru
        self.start_measurement_button = QPushButton("Rozpocznij pomiar")
        self.start_measurement_button.clicked.connect(self.start_measurement)

        # Przycisk do zapisywania próbki
        save_sample_button = QPushButton("Zapisz próbkę")
        save_sample_button.clicked.connect(self.save_sample)

        # Dodanie przycisków do zakładki
        measurement_layout.addWidget(self.start_measurement_button)
        measurement_layout.addWidget(save_sample_button)

        # Dodanie konsoli z wartościami pomiaru
        console_label = QLabel("Konsola z wartościami pomiaru:")
        measurement_layout.addWidget(console_label)

        # Pole tekstowe do wyświetlania wartości pomiaru
        self.measurement_console = QLineEdit()
        self.measurement_console.setReadOnly(True)
        measurement_layout.addWidget(self.measurement_console)

        measurement_tab.setLayout(measurement_layout)

        # Dodanie zakładki do QTabWidget
        self.tabs.addTab(measurement_tab, "Pomiar bez maszyny")

    def open_measurement_for_machine(self, machine_number):
        if not self.data_loaded:
            print("Błąd: Nie załadowano pliku!")
            return
        
        if machine_number < 1 or machine_number > len(self.machines):
            print("Błąd: Nieprawidłowy numer maszyny!")
            return
        
        # Tworzymy nową zakładkę do pomiaru dla maszyny
        measurement_tab = QWidget()  # To jest nowa zakładka
        measurement_layout = QVBoxLayout()

        # Dodanie etykiety z informacją o pomiarze dla maszyny
        measurement_label = QLabel(f"Pomiary dla maszyny: {self.machines[machine_number - 1]['name']}")
        measurement_layout.addWidget(measurement_label)

        # Dodanie etykiety z czasem trwania pomiaru
        self.duration_label = QLabel("Czas trwania pomiaru: 0s")
        measurement_layout.addWidget(self.duration_label)

        # Przycisk do rozpoczęcia pomiaru
        self.start_measurement_button = QPushButton("Rozpocznij pomiar")
        self.start_measurement_button.clicked.connect(self.start_measurement)

        # Przycisk do zapisywania próbki
        save_sample_button = QPushButton("Zapisz próbkę")
        save_sample_button.clicked.connect(self.save_sample)

        # Dodanie przycisków do zakładki
        measurement_layout.addWidget(self.start_measurement_button)
        measurement_layout.addWidget(save_sample_button)

        # Dodanie konsoli z wartościami pomiaru
        console_label = QLabel("Konsola z wartościami pomiaru:")
        measurement_layout.addWidget(console_label)

        # Pole tekstowe do wyświetlania wartości pomiaru
        self.measurement_console = QLineEdit()
        self.measurement_console.setReadOnly(True)
        measurement_layout.addWidget(self.measurement_console)

        measurement_tab.setLayout(measurement_layout)

        # Dodanie zakładki do QTabWidget
        self.tabs.addTab(measurement_tab, f"Pomiar maszyna {machine_number}")

    def start_measurement(self):
        # Akcja do rozpoczęcia pomiaru
        print("Pomiar rozpoczęty")
        self.measurement_time = 0
        self.timer.start(1000)  # Timer odlicza co 1 sekundę
        self.start_measurement_button.setEnabled(False)
        self.update_measurement_console()
        self.generate_plot()  # Generowanie wykresu po rozpoczęciu pomiaru

    def update_measurement_time(self):
        self.measurement_time += 1
        self.duration_label.setText(f"Czas trwania pomiaru: {self.measurement_time}s")
        self.update_measurement_console()  # Aktualizacja konsoli przy każdej zmianie czasu

    def update_measurement_console(self):
        if self.df is not None:
            # Symulacja wartości pomiaru
            measurement_value = np.random.random()  # Losowa wartość pomiaru
            self.measurement_console.setText(f"Wartość pomiaru: {measurement_value:.2f}")

    def save_sample(self):
        # Akcja do zapisywania próbki
        print("Próbka zapisana")
        self.timer.stop()
        self.start_measurement_button.setEnabled(True)

    def generate_plot(self):
        if self.df is not None:
            print("Generowanie wykresu...")

            # Tworzymy wykres na podstawie danych
            plt.figure()
            plt.plot(self.df.iloc[:, 0])  # Używamy tylko pierwszej kolumny danych
            plt.title("Wykres pomiaru drgań")
            plt.xlabel("Próbka")
            plt.ylabel("Amplituda")
            plt.grid(True)
            plt.show()

    def open_analyze_samples(self):
        # Tworzymy nową zakładkę do analizy próbek
        analyze_tab = QWidget()
        analyze_layout = QVBoxLayout()

        # Dodanie etykiety z informacją o analizie próbek
        analyze_label = QLabel("Analiza próbek")
        analyze_layout.addWidget(analyze_label)

        # Wybierz maszynę
        machine_selection_label = QLabel("Wybierz maszynę:")
        analyze_layout.addWidget(machine_selection_label)

        self.machine_selection_combo = QComboBox()
        for machine in self.machines:
            self.machine_selection_combo.addItem(machine['name'])
        analyze_layout.addWidget(self.machine_selection_combo)

        # Wybierz próbkę (plik)
        sample_selection_label = QLabel("Wybierz próbkę (plik):")
        analyze_layout.addWidget(sample_selection_label)

        self.sample_selection_button = QPushButton("Wybierz plik z próbką")
        self.sample_selection_button.clicked.connect(self.select_sample_file)
        analyze_layout.addWidget(self.sample_selection_button)

        # Wybierz model (algorytm analizy)
        model_selection_label = QLabel("Wybierz model (algorytm analizy):")
        analyze_layout.addWidget(model_selection_label)

        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItems(["Model 1: Analiza podstawowa", "Model 2: Analiza zaawansowana", "Model 3: Analiza statystyczna"])  # Przykładowe modele
        analyze_layout.addWidget(self.model_selection_combo)

        # Przycisk do rozpoczęcia analizy
        analyze_button = QPushButton("Rozpocznij analizę")
        analyze_button.clicked.connect(self.start_analysis)
        analyze_layout.addWidget(analyze_button)

        # Sekcja wyników analizy
        self.results_label = QLabel("Wyniki analizy:")
        analyze_layout.addWidget(self.results_label)

        # Pole tekstowe do wyświetlania wyników analizy
        self.results_console = QLineEdit()
        self.results_console.setReadOnly(True)
        analyze_layout.addWidget(self.results_console)

        # Wykres wyników analizy
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        analyze_layout.addWidget(self.canvas)

        analyze_tab.setLayout(analyze_layout)

        # Dodanie zakładki do QTabWidget
        self.tabs.addTab(analyze_tab, "Analiza próbek")

    def select_sample_file(self):
        # Funkcja do wyboru pliku z próbką
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik z próbką", "", "Pliki CSV (*.csv)")
        if file_path:
            print(f"Wybrano plik z próbką: {file_path}")
            # Tutaj można dodać logikę ładowania i przetwarzania pliku z próbką

    def start_analysis(self):
        # Akcja do rozpoczęcia analizy próbek
        selected_machine = self.machine_selection_combo.currentText()
        selected_model = self.model_selection_combo.currentText()

        # Symulacja wyników analizy
        self.results_console.setText(f"Analiza dla maszyny {selected_machine} z modelem {selected_model} zakończona pomyślnie.")

        # Generowanie wykresu wyników analizy
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 4])  # Przykładowe dane do wykresu
        ax.set_title("Wyniki analizy")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Amplituda")
        self.canvas.draw()

# Uruchomienie aplikacji
if __name__ == "__main__": 
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()