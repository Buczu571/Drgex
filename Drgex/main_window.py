import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox,  QPushButton, QFileDialog, QLabel, QLineEdit, QTabWidget, QGridLayout, QComboBox
from PyQt6.QtCore import QTimer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from functools import partial
from add_machine_window import AddMachineWindow
from machine_manager import load_machines_from_folder, delete_machine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitorowanie Drgań Maszyn")
        self.setGeometry(100, 100, 800, 600)
        self.data_loaded = False
        self.df = None
        self.measurement_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_measurement_time)
        self.machines = []
        self.machine_labels = []
        self.measure_buttons = []

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.measure_without_machine_button = QPushButton("Pomiar bez wybranej maszyny")
        self.file_samples_button = QPushButton("Plik z próbkami")
        self.edit_file_button = QPushButton("Edytuj maszynę")
        self.add_machine_button = QPushButton("Dodaj nową maszynę")
        self.analyze_samples_button = QPushButton("Analiza próbek")

        self.grid_layout.addWidget(self.measure_without_machine_button, 0, 0)
        self.grid_layout.addWidget(self.file_samples_button, 0, 3)
        self.grid_layout.addWidget(self.edit_file_button, 0, 4)
        self.grid_layout.addWidget(self.add_machine_button, 1, 0)
        self.grid_layout.addWidget(self.analyze_samples_button, 1, 1)

        self.main_layout.addLayout(self.grid_layout)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        central_widget.setLayout(self.main_layout)

        self.add_machine_button.clicked.connect(self.open_add_machine_window)
        self.measure_without_machine_button.clicked.connect(self.open_measurement_without_machine)
        self.file_samples_button.clicked.connect(self.load_file)
        self.analyze_samples_button.clicked.connect(self.open_analyze_samples)  

        self.machines = load_machines_from_folder()
        self.update_machine_ui()

    def open_add_machine_window(self):
        self.add_machine_window = AddMachineWindow(self)
        self.add_machine_window.show()

    def update_measurement_time(self):
        self.measurement_time += 1
        self.duration_label.setText(f"Czas trwania pomiaru: {self.measurement_time}s")
        self.update_measurement_console()

    def update_measurement_console(self):
        if self.df is not None:
            measurement_value = np.random.random()
            self.measurement_console.setText(f"Wartość pomiaru: {measurement_value:.2f}")

    def update_machine_ui(self):
        for label in self.machine_labels:
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        for button in self.measure_buttons:
            self.grid_layout.removeWidget(button)
            button.deleteLater()

        self.machine_labels.clear()
        self.measure_buttons.clear()

        for i, machine in enumerate(self.machines):
            machine_label = QLabel(f"Maszyna {i + 1}: {machine['name']}")
            self.machine_labels.append(machine_label)

            measure_button = QPushButton(f"Pomiar dla maszyny {i + 1}")
            measure_button.clicked.connect(partial(self.open_measurement_for_machine, i + 1))
            self.measure_buttons.append(measure_button)

            edit_button = QPushButton("Edytuj maszynę")
            edit_button.clicked.connect(partial(self.edit_machine, i))
            self.measure_buttons.append(edit_button)

            delete_button = QPushButton("Usuń maszynę")
            delete_button.clicked.connect(partial(self.delete_machine, i))
            self.measure_buttons.append(delete_button)

            sample_info_button = QPushButton("Próbki")
            sample_info_button.clicked.connect(partial(self.show_samples, i))
            self.measure_buttons.append(sample_info_button)

            self.grid_layout.addWidget(machine_label, i + 2, 1)
            self.grid_layout.addWidget(measure_button, i + 2, 2)
            self.grid_layout.addWidget(edit_button, i + 2, 3)
            self.grid_layout.addWidget(delete_button, i + 2, 4)
            self.grid_layout.addWidget(sample_info_button, i + 2, 5)

    def delete_machine(self, machine_index):
        delete_machine(machine_index, self.machines)
        self.update_machine_ui()

    def edit_machine(self, machine_index):
        machine_name = self.machines[machine_index]['name']
        folder_name = "machines"
        file_path = os.path.join(folder_name, f"{machine_name}.txt")
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            QMessageBox.warning(self, "Błąd", f"Plik '{file_path}' nie istnieje.")

    def add_machine(self, name, technical_data, location, additional_info):
        self.machines.append({
            "name": name,
            "technical_data": technical_data,
            "location": location,
            "additional_info": additional_info
        })
        self.update_machine_ui()

    def load_file(self):
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
        
        measurement_tab = QWidget()
        measurement_layout = QVBoxLayout()
        measurement_label = QLabel("Pomiar bez wybranej maszyny")
        measurement_layout.addWidget(measurement_label)
        self.duration_label = QLabel("Czas trwania pomiaru: 0s")
        measurement_layout.addWidget(self.duration_label)
        self.start_measurement_button = QPushButton("Rozpocznij pomiar")
        self.start_measurement_button.clicked.connect(self.start_measurement)
        save_sample_button = QPushButton("Zapisz próbkę")
        save_sample_button.clicked.connect(self.save_sample)
        measurement_layout.addWidget(self.start_measurement_button)
        measurement_layout.addWidget(save_sample_button)
        console_label = QLabel("Konsola z wartościami pomiaru:")
        measurement_layout.addWidget(console_label)
        self.measurement_console = QLineEdit()
        self.measurement_console.setReadOnly(True)
        measurement_layout.addWidget(self.measurement_console)
        measurement_tab.setLayout(measurement_layout)
        self.tabs.addTab(measurement_tab, "Pomiar bez maszyny")

    def open_measurement_for_machine(self, machine_number):
        if not self.data_loaded:
            print("Błąd: Nie załadowano pliku!")
            return
        
        if machine_number < 1 or machine_number > len(self.machines):
            print("Błąd: Nieprawidłowy numer maszyny!")
            return
        
        measurement_tab = QWidget()
        measurement_layout = QVBoxLayout()
        measurement_label = QLabel(f"Pomiary dla maszyny: {self.machines[machine_number - 1]['name']}")
        measurement_layout.addWidget(measurement_label)
        self.duration_label = QLabel("Czas trwania pomiaru: 0s")
        measurement_layout.addWidget(self.duration_label)
        self.start_measurement_button = QPushButton("Rozpocznij pomiar")
        self.start_measurement_button.clicked.connect(self.start_measurement)
        save_sample_button = QPushButton("Zapisz próbkę")
        save_sample_button.clicked.connect(partial(self.save_sample_for_machine, machine_number - 1))
        measurement_layout.addWidget(self.start_measurement_button)
        measurement_layout.addWidget(save_sample_button)
        console_label = QLabel("Konsola z wartościami pomiaru:")
        measurement_layout.addWidget(console_label)
        self.measurement_console = QLineEdit()
        self.measurement_console.setReadOnly(True)
        measurement_layout.addWidget(self.measurement_console)
        measurement_tab.setLayout(measurement_layout)
        self.tabs.addTab(measurement_tab, f"Pomiar maszyna {machine_number}")

    def start_measurement(self):
        print("Pomiar rozpoczęty")
        self.measurement_time = 0
        self.timer.start(1000)
        self.start_measurement_button.setEnabled(False)
        self.update_measurement_console()
        self.generate_plot()

    def save_sample_for_machine(self, machine_index):
        if self.df is not None:
            machine_name = self.machines[machine_index]['name']
            folder_name = os.path.join("machines", machine_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            sample_number = len(os.listdir(folder_name)) + 1
            file_path = os.path.join(folder_name, f"próbka_{sample_number}.csv")
            self.df.to_csv(file_path, index=False)
            print(f"Próbka zapisana dla maszyny {machine_name} w pliku: {file_path}")
            self.timer.stop()
            self.start_measurement_button.setEnabled(True)

    def save_sample(self):
        if self.df is not None:
            folder_name = "samples_without_machine"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            sample_number = len(os.listdir(folder_name)) + 1
            file_path = os.path.join(folder_name, f"próbka_{sample_number}.csv")
            self.df.to_csv(file_path, index=False)
            print(f"Próbka zapisana bez przypisania do maszyny w pliku: {file_path}")
            self.timer.stop()
            self.start_measurement_button.setEnabled(True)

    def generate_plot(self):
        if self.df is not None:
            print("Generowanie wykresu...")
            plt.figure()
            plt.plot(self.df.iloc[:, 0])
            plt.title("Wykres pomiaru drgań")
            plt.xlabel("Próbka")
            plt.ylabel("Amplituda")
            plt.grid(True)
            plt.show()

    def show_samples(self, machine_index):
        machine_name = self.machines[machine_index]['name']
        folder_name = os.path.join("machines", machine_name)
        if os.path.exists(folder_name):
            sample_files = os.listdir(folder_name)
            if sample_files:
                sample_text = "\n".join(sample_files)
                QMessageBox.information(self, f"Próbki dla maszyny {machine_name}", sample_text)
            else:
                QMessageBox.information(self, f"Próbki dla maszyny {machine_name}", "Brak zapisanych próbek.")
        else:
            QMessageBox.information(self, f"Próbki dla maszyny {machine_name}", "Brak zapisanych próbek.")

    def open_analyze_samples(self):
        analyze_tab = QWidget()
        analyze_layout = QVBoxLayout()
        analyze_label = QLabel("Analiza próbek")
        analyze_layout.addWidget(analyze_label)
        machine_selection_label = QLabel("Wybierz maszynę:")
        analyze_layout.addWidget(machine_selection_label)
        self.machine_selection_combo = QComboBox()
        for machine in self.machines:
           self.machine_selection_combo.addItem(machine['name'])
        analyze_layout.addWidget(self.machine_selection_combo)
        sample_selection_label = QLabel("Wybierz próbkę (plik):")
        analyze_layout.addWidget(sample_selection_label)
        self.sample_selection_button = QPushButton("Wybierz plik z próbką")
        self.sample_selection_button.clicked.connect(self.select_sample_file)
        analyze_layout.addWidget(self.sample_selection_button)
        model_selection_label = QLabel("Wybierz model (algorytm analizy):")
        analyze_layout.addWidget(model_selection_label)
        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItems(["Model 1: Analiza podstawowa", "Model 2: Analiza zaawansowana", "Model 3: Analiza statystyczna"])
        analyze_layout.addWidget(self.model_selection_combo)
        analyze_button = QPushButton("Rozpocznij analizę")
        analyze_button.clicked.connect(self.start_analysis)
        analyze_layout.addWidget(analyze_button)
        self.results_label = QLabel("Wyniki analizy:")
        analyze_layout.addWidget(self.results_label)
        self.results_console = QLineEdit()
        self.results_console.setReadOnly(True)
        analyze_layout.addWidget(self.results_console)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        analyze_layout.addWidget(self.canvas)
        analyze_tab.setLayout(analyze_layout)
        self.tabs.addTab(analyze_tab, "Analiza próbek")

    def select_sample_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik z próbką", "", "Pliki CSV (*.csv)")
        if file_path:
            print(f"Wybrano plik z próbką: {file_path}")

    def start_analysis(self):
        selected_machine = self.machine_selection_combo.currentText()
        selected_model = self.model_selection_combo.currentText()
        self.results_console.setText(f"Analiza dla maszyny {selected_machine} z modelem {selected_model} zakończona pomyślnie.")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 4])
        ax.set_title("Wyniki analizy")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Amplituda")
        self.canvas.draw()

if __name__ == "__main__": 
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()