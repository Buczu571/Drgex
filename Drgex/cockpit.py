import sys
import os
import shutil
import csv
import numpy
import string
from scipy.signal import savgol_filter
from PyQt6.QtWidgets import QApplication, QPushButton, QMainWindow, QSpacerItem, QSizePolicy, QGridLayout, QWidget, QLineEdit, QMessageBox, QVBoxLayout, QTableWidget
from serialreader import SerialReader
from dialogs import NewMachineDialog, EditMachineDialog, SelectMachineDialog, NewMeasurmentsDialog, EditMeasurmentsDialog, SelectMeasurmentsDialog, SelectAnalyzeDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

class Cockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kokpit")
        self.setGeometry(0, 0, 1536, 864)
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)


        self.machine_list = []
        self.selected_machine = None
        self.selected_machine_intr = "Wybrane urządzenie: "

        self.measurments_list = []
        self.selected_measurments = None
        self.selected_measurments_intr = "Wybrany zestaw pomiarów: "

        self.csv_standard_length = None

        self.fft1_average = None
        self.fft2_average = None
        self.xf1 = None
        self.xf2 = None

        self.fftdiff = None

        self.analyze_selector = 0
        self.analyze_folders = None

        self.grid_layout = QGridLayout()

        for row in range(32):
            for col in range(32):
                window_width = self.width()
                window_height = self.height()

                spacer_width = int(window_width / 32)
                spacer_height = int(window_height / 32)

                spacer = QSpacerItem(spacer_width, spacer_height, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                
                self.grid_layout.addItem(spacer, row, col)

        self.select_machine_btn = QPushButton("Wybierz urządzenie")
        self.select_machine_btn.clicked.connect(self.select_machine)
        self.grid_layout.addWidget(self.select_machine_btn, 0, 0, 1, 4)

        self.new_machine_btn = QPushButton("Dodaj urządzenie")
        self.new_machine_btn.clicked.connect(self.new_machine)
        self.grid_layout.addWidget(self.new_machine_btn, 1, 0, 1, 4)

        self.edit_machine_btn = QPushButton("Edytuj urządzenie")
        self.edit_machine_btn.clicked.connect(self.edit_machine)
        self.grid_layout.addWidget(self.edit_machine_btn, 2, 0, 1, 4)

        self.delete_machine_btn = QPushButton("Usuń urządzenie")
        self.delete_machine_btn.clicked.connect(self.delete_machine)
        self.grid_layout.addWidget(self.delete_machine_btn, 3, 0, 1, 4)




        self.select_measurments_btn = QPushButton("Wybierz zestaw pomiarów")
        self.select_measurments_btn.clicked.connect(self.select_measurments)
        self.grid_layout.addWidget(self.select_measurments_btn, 0, 4, 1, 4)

        self.new_measurments_btn = QPushButton("Dodaj zestaw pomiarów")
        self.new_measurments_btn.clicked.connect(self.new_measurments)
        self.grid_layout.addWidget(self.new_measurments_btn, 1, 4, 1, 4)

        self.edit_measurments_btn = QPushButton("Edytuj zestaw pomiarów")
        self.edit_measurments_btn.clicked.connect(self.edit_measurments)
        self.grid_layout.addWidget(self.edit_measurments_btn, 2, 4, 1, 4)

        self.delete_measurments_btn = QPushButton("Usuń zestaw pomiarów")
        self.delete_measurments_btn.clicked.connect(self.delete_measurments)
        self.grid_layout.addWidget(self.delete_measurments_btn, 3, 4, 1, 4)

        self.selected_machine_field = QLineEdit(self)
        self.selected_machine_field.setReadOnly(True)
        self.selected_machine_field.setText(self.selected_machine_intr)


        self.selected_machine_field.setStyleSheet("""
            background-color: transparent;
            border: none;
        """)

        self.grid_layout.addWidget(self.selected_machine_field, 0, 24, 1, 8)

        self.selected_measurments_field = QLineEdit(self)
        self.selected_measurments_field.setReadOnly(True)
        self.selected_measurments_field.setText(self.selected_measurments_intr)

        self.selected_measurments_field.setStyleSheet("""
            background-color: transparent;
            border: none;
        """)

        self.grid_layout.addWidget(self.selected_measurments_field, 1, 24, 1, 8)

        self.serialreader_btn = QPushButton("Uruchom pomiar")
        self.serialreader_btn.clicked.connect(self.open_serialreader)
        self.grid_layout.addWidget(self.serialreader_btn, 2, 24, 1, 8)

        self.analyze_btn = QPushButton("Analiza porównawcza")
        self.analyze_btn.clicked.connect(self.analyze)
        self.grid_layout.addWidget(self.analyze_btn, 3, 24, 1, 6)

        self.analyze_mode_btn = QPushButton("Tryb analizy")
        self.analyze_mode_btn.clicked.connect(self.analyze_mode_selector)
        self.grid_layout.addWidget(self.analyze_mode_btn, 3, 30, 1, 2)

        central_widget.setLayout(self.grid_layout)


    def select_machine(self):
        self.machine_list = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)

            if os.path.isdir(folder_path) and folder != "__pycache__":
                self.machine_list.append(folder)

        dialog = SelectMachineDialog(self, self.machine_list)

        if dialog.exec():
            self.selected_machine = dialog.get_machine_name()
            self.selected_machine_field.setText(self.selected_machine_intr + self.selected_machine)
            self.selected_measurments = None
            self.selected_measurments_field.setText(self.selected_measurments_intr)


    def new_machine(self):
        dialog = NewMachineDialog(self)
        if dialog.exec():
            machine_name = dialog.get_machine_name()

            if not machine_name:
                QMessageBox.warning(self, "Błąd", "Nazwa nie może być pusta!")
                return
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            machine_folder = os.path.join(base_dir, machine_name)

            if os.path.exists(machine_folder):
                QMessageBox.warning(self, "Błąd", "Urządzenie o takiej nazwie już istnieje!")
            else:
                os.makedirs(machine_folder)
                self.selected_machine = machine_name
                self.selected_machine_field.setText(self.selected_machine_intr + self.selected_machine)
                self.selected_measurments = None
                self.selected_measurments_field.setText(self.selected_measurments_intr)
                QMessageBox.information(self, "Sukces", f"Urządzenie '{machine_name}' zostało dodane!")


    def edit_machine(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        dialog = EditMachineDialog(self, self.selected_machine)
        if dialog.exec():
            machine_name = dialog.get_machine_name()

            if not machine_name:
                QMessageBox.warning(self, "Błąd", "Nazwa nie może być pusta!")
                return
            
            if machine_name == self.selected_machine:
                QMessageBox.warning(self, "Błąd", "Nazwa nie została zmieniona!")
                return
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            machine_folder = os.path.join(base_dir, machine_name)
            old_machine_folder = os.path.join(base_dir, self.selected_machine)

            if os.path.exists(machine_folder):
                QMessageBox.warning(self, "Błąd", "Urządzenie o takiej nazwie już istnieje!")
            else:
                os.rename(old_machine_folder, machine_folder)
                self.selected_machine = machine_name
                self.selected_machine_field.setText(self.selected_machine_intr + self.selected_machine)
                self.selected_measurments = None
                self.selected_measurments_field.setText(self.selected_measurments_intr)
                QMessageBox.information(self, "Sukces", f"Nazwa urządzenia '{machine_name}' została zmieniona!")


    def delete_machine(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        machine_folder = os.path.join(base_dir, self.selected_machine)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Potwierdzenie")
        msg_box.setText(f"Czy na pewno chcesz usunąć urządzenie '{self.selected_machine}'?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        yes_button = msg_box.addButton("Tak", QMessageBox.ButtonRole.AcceptRole)
        no_button = msg_box.addButton("Nie", QMessageBox.ButtonRole.RejectRole)

        msg_box.exec()

        if msg_box.clickedButton() == yes_button:
            try:
                shutil.rmtree(machine_folder)
                old_machine = self.selected_machine
                self.selected_machine = None
                self.selected_machine_field.setText(self.selected_machine_intr) 
                self.selected_measurments = None
                self.selected_measurments_field.setText(self.selected_measurments_intr)
                QMessageBox.information(self, "Sukces", f"Urządzenie '{old_machine}' zostało usunięte.")
            except Exception as e:
                QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć folderu: {e}")


    def select_measurments(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return

        self.measurments_list = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        machine_folder = os.path.join(base_dir, self.selected_machine)
        for folder in os.listdir(machine_folder):
            self.measurments_list.append(folder)

        dialog = SelectMeasurmentsDialog(self, self.measurments_list)

        if dialog.exec():
            self.selected_measurments = dialog.get_measurments_name()
            self.selected_measurments_field.setText(self.selected_measurments_intr + self.selected_measurments)


    def new_measurments(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        dialog = NewMeasurmentsDialog(self)
        if dialog.exec():
            measurments_name = dialog.get_measurments_name()

            if not measurments_name:
                QMessageBox.warning(self, "Błąd", "Nazwa nie może być pusta!")
                return
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            measurments_folder = os.path.join(base_dir, self.selected_machine, measurments_name)

            if os.path.exists(measurments_folder):
                QMessageBox.warning(self, "Błąd", "Zestaw pomiarów o takiej nazwie już istnieje!")
            else:
                os.makedirs(measurments_folder)
                self.selected_measurments = measurments_name
                self.selected_measurments_field.setText(self.selected_measurments_intr + self.selected_measurments)
                QMessageBox.information(self, "Sukces", f"Zestaw pomiarów '{measurments_name}' został dodany!")

    
    def edit_measurments(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        if self.selected_measurments == None:
            QMessageBox.warning(self, "Błąd", "Zestaw pomiarów nie został wybrany!")
            return
        
        dialog = EditMeasurmentsDialog(self, self.selected_measurments)
        if dialog.exec():
            measurments_name = dialog.get_measurments_name()

            if not measurments_name:
                QMessageBox.warning(self, "Błąd", "Nazwa nie może być pusta!")
                return
            
            if measurments_name == self.selected_measurments:
                QMessageBox.warning(self, "Błąd", "Nazwa nie została zmieniona!")
                return
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            measurments_folder = os.path.join(base_dir, self.selected_machine, measurments_name)
            old_measurments_folder = os.path.join(base_dir, self.selected_machine, self.selected_measurments)

            if os.path.exists(measurments_folder):
                QMessageBox.warning(self, "Błąd", "Urządzenie o takiej nazwie już istnieje!")
            else:
                os.rename(old_measurments_folder, measurments_folder)
                self.selected_measurments = measurments_name
                self.selected_measurments_field.setText(self.selected_measurments_intr + self.selected_measurments)
                QMessageBox.information(self, "Sukces", f"Nazwa zestawu pomiarów '{measurments_folder}' została zmieniona!")

    
    def delete_measurments(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        if self.selected_measurments == None:
            QMessageBox.warning(self, "Błąd", "Zestaw pomiarów nie został wybrany!")
            return
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        measurments_folder = os.path.join(base_dir, self.selected_machine, self.selected_measurments)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Potwierdzenie")
        msg_box.setText(f"Czy na pewno chcesz usunąć zestaw pomiarów '{self.selected_measurments}'?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        yes_button = msg_box.addButton("Tak", QMessageBox.ButtonRole.AcceptRole)
        no_button = msg_box.addButton("Nie", QMessageBox.ButtonRole.RejectRole)

        msg_box.exec()

        if msg_box.clickedButton() == yes_button:
            try:
                shutil.rmtree(measurments_folder)
                old_measurments = self.selected_measurments
                self.selected_measurments = None
                self.selected_measurments_field.setText(self.selected_measurments_intr) 
                QMessageBox.information(self, "Sukces", f"Zestaw pomiarów '{old_measurments}' został usunięty.")
            except Exception as e:
                QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć folderu: {e}")


    def open_serialreader(self):
        self.new_window = SerialReader(self.selected_machine, self.selected_measurments)
        self.new_window.show()

    def analyze(self):
        if self.selected_machine == None:
            QMessageBox.warning(self, "Błąd", "Urządzenie nie zostało wybrane!")
            return
        
        self.measurments_list = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        machine_folder = os.path.join(base_dir, self.selected_machine)
        for folder in os.listdir(machine_folder):
            self.measurments_list.append(folder)

        dialog = SelectAnalyzeDialog(self, self.measurments_list)

        if dialog.exec():
            self.analyze_folders = dialog.get_analyze_name()

            analyze_folder1 = os.path.join(machine_folder, self.analyze_folders[0])
            csv_files1 = [f for f in os.listdir(analyze_folder1) if f.endswith(".csv")]

            analyze_folder2 = os.path.join(machine_folder, self.analyze_folders[1])
            csv_files2 = [f for f in os.listdir(analyze_folder2) if f.endswith(".csv")]

            csv_lengths1 = []
            csv_lengths2 = []

            for file_name in csv_files1:
                with open(os.path.join(analyze_folder1, file_name), mode="r", newline="") as file:
                    reader = csv.reader(file)
                    #MT
                    csv_lengths1.append(int(next(reader)[0]))
            
            for file_name in csv_files2:
                with open(os.path.join(analyze_folder2, file_name), mode="r", newline="") as file:
                    reader = csv.reader(file)
                    csv_lengths2.append(int(next(reader)[0]))

            self.csv_standard_length = min(min(csv_lengths1), min(csv_lengths2))

            fft_magnitude_sum1 = numpy.zeros(self.csv_standard_length)
            fft_magnitude_sum2 = numpy.zeros(self.csv_standard_length)

            for file_name in csv_files1:
                with open(os.path.join(analyze_folder1, file_name), mode="r", newline="") as file:
                    reader = csv.reader(file)
                    import_data = [int(row[0]) for i, row in enumerate(reader) if i <= self.csv_standard_length]  

                    import_data.pop(0)

                    import_data = [(i / 4096) * 3.3 for i in import_data]
                    import_data = numpy.array(import_data) - numpy.mean(import_data)

                    #window = numpy.hanning(self.csv_standard_length)
                    #import_data = import_data * window

                    yf = numpy.fft.fft(import_data)
                    fft_magnitude_sum1 += numpy.abs(yf) ** 2

            self.fft1_average = fft_magnitude_sum1/self.csv_standard_length
            self.xf1 = numpy.fft.fftfreq(self.csv_standard_length, 1/self.csv_standard_length)


            for file_name in csv_files2:
                with open(os.path.join(analyze_folder2, file_name), mode="r", newline="") as file:
                    reader = csv.reader(file)
                    import_data = [int(row[0]) for i, row in enumerate(reader) if i <= self.csv_standard_length]  

                    import_data.pop(0)

                    import_data = [(i / 4096) * 3.3 for i in import_data]
                    import_data = numpy.array(import_data) - numpy.mean(import_data)

                    #window = numpy.hanning(self.csv_standard_length)
                    #import_data = import_data * window

                    yf = numpy.fft.fft(import_data)
                    fft_magnitude_sum2 += numpy.abs(yf) ** 2

            
            self.fft2_average = fft_magnitude_sum2/self.csv_standard_length
            self.xf2 = numpy.fft.fftfreq(self.csv_standard_length, 1/self.csv_standard_length)        

            self.analyze_selector = 0

            self.analyze_mode3()
            self.analyze_mode4()
            
            self.analyze_mode1()
            self.analyze_mode2()

            self.analyze_selector = 1

            


    def analyze_mode1(self):
        if self.analyze_selector == 0 or self.analyze_selector == 2:
            if hasattr(self, "plot_fft1_widget"):
                self.plot_fft1_canvas.figure.clf()

            if hasattr(self, "plot_fft1_layout"):
                self.plot_fft1_layout.setParent(None)
            
            if hasattr(self, "fft1_toolbar"):
                self.fft1_toolbar.setParent(None)

            self.plot_fft1_widget = QWidget(self)
            self.plot_fft1_layout = QVBoxLayout(self.plot_fft1_widget)

            self.plot_fft1_canvas = FigureCanvas(Figure(figsize=(5, 3)))

            self.fft1_toolbar = NavigationToolbar2QT(self.plot_fft1_canvas, self)
            self.plot_fft1_layout.addWidget(self.fft1_toolbar)

            self.plot_fft1_layout.addWidget(self.plot_fft1_canvas)

            self.grid_layout.addWidget(self.plot_fft1_widget, 4, 0, 28, 16)

            ax = self.plot_fft1_canvas.figure.add_subplot(111)

            ax.clear()
            ax.plot(self.xf1[:self.csv_standard_length//2], self.fft1_average[:self.csv_standard_length//2], marker='o', linestyle='-', markersize=2)

            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Amplituda")
            ax.set_title("Analiza FFT - " + self.analyze_folders[0])

            self.plot_fft1_canvas.draw()
        
        elif self.analyze_selector == 1:
            if hasattr(self, "plot_fft1_widget"):
                self.plot_fft1_widget.setParent(None)

            #if hasattr(self, "plot_fft1_canvas"):
            #    self.plot_fft1_canvas.setParent(None)

            #if hasattr(self, "plot_fft1_layout"):
            #    self.plot_fft1_layout.setParent(None)
            
            #if hasattr(self, "fft1_toolbar"):
            #    self.fft1_toolbar.setParent(None)



    def analyze_mode2(self):
        if self.analyze_selector == 0 or self.analyze_selector == 2:
            if hasattr(self, "plot_fft2_widget"):
                self.plot_fft2_canvas.figure.clf()

            if hasattr(self, "plot_fft2_layout"):
                self.plot_fft2_layout.setParent(None)
            
            if hasattr(self, "fft2_toolbar"):
                self.fft2_toolbar.setParent(None)

            self.plot_fft2_widget = QWidget(self)
            self.plot_fft2_layout = QVBoxLayout(self.plot_fft2_widget)

            self.plot_fft2_canvas = FigureCanvas(Figure(figsize=(5, 3)))

            self.fft2_toolbar = NavigationToolbar2QT(self.plot_fft2_canvas, self)
            self.plot_fft2_layout.addWidget(self.fft2_toolbar)

            self.plot_fft2_layout.addWidget(self.plot_fft2_canvas)

            self.grid_layout.addWidget(self.plot_fft2_widget, 4, 16, 28, 16)

            ax = self.plot_fft2_canvas.figure.add_subplot(111)

            ax.clear()
            ax.plot(self.xf2[:self.csv_standard_length//2], self.fft2_average[:self.csv_standard_length//2], marker='o', linestyle='-', markersize=2)

            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Amplituda")
            ax.set_title("Analiza FFT - " + self.analyze_folders[1])

            self.plot_fft2_canvas.draw()

        elif self.analyze_selector == 1:
            if hasattr(self, "plot_fft2_widget"):
                self.plot_fft2_widget.setParent(None)

            #if hasattr(self, "plot_fft2_canvas"):
            #    self.plot_fft2_canvas.setParent(None)

            #if hasattr(self, "plot_fft2_layout"):
            #    self.plot_fft2_layout.setParent(None)
            
            #if hasattr(self, "fft2_toolbar"):
            #    self.fft2_toolbar.setParent(None)


    def analyze_mode3(self):
        if self.analyze_selector == 1:
            if hasattr(self, "plot_fft3_widget"):
                self.plot_fft3_canvas.figure.clf()

            if hasattr(self, "plot_fft3_layout"):
                self.plot_fft3_layout.setParent(None)
            
            if hasattr(self, "fft3_toolbar"):
                self.fft3_toolbar.setParent(None)

            self.plot_fft3_widget = QWidget(self)
            self.plot_fft3_layout = QVBoxLayout(self.plot_fft3_widget)

            self.plot_fft3_canvas = FigureCanvas(Figure(figsize=(5, 3)))

            self.fft3_toolbar = NavigationToolbar2QT(self.plot_fft3_canvas, self)
            self.plot_fft3_layout.addWidget(self.fft3_toolbar)

            self.plot_fft3_layout.addWidget(self.plot_fft3_canvas)

            self.grid_layout.addWidget(self.plot_fft3_widget, 4, 0, 28, 16)

            ax = self.plot_fft3_canvas.figure.add_subplot(111)

            self.fftdiff = numpy.abs(self.fft1_average - self.fft2_average)
            self.fftdiff = savgol_filter(self.fftdiff, window_length=7, polyorder=2)
            self.fftdiff = numpy.clip(self.fftdiff, 0, None)

            ax.clear()
            ax.plot(self.xf2[:self.csv_standard_length//2], self.fftdiff[:self.csv_standard_length//2], marker='o', linestyle='-', markersize=2)

            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Amplituda")
            ax.set_title("Różnica amplitud FFT - " + self.analyze_folders[0] + " | " + self.analyze_folders[1])

            self.plot_fft3_canvas.draw()

        elif self.analyze_selector == 0 or self.analyze_selector == 2:
            if hasattr(self, "plot_fft3_widget"):
                self.plot_fft3_widget.setParent(None)

            #if hasattr(self, "plot_fft3_canvas"):
            #    self.plot_fft3_canvas.setParent(None)

            #if hasattr(self, "plot_fft3_layout"):
            #    self.plot_fft3_layout.setParent(None)
            
            #if hasattr(self, "fft3_toolbar"):
            #    self.fft3_toolbar.setParent(None)

    
    def analyze_mode4(self):
        if self.analyze_selector == 1:

            self.line_edits = []
            self.line_edits_all = []

            self.line_freq = QLineEdit(self)
            self.line_freq.setReadOnly(True)
            self.line_freq.setText("Częstotliwość")
            self.line_edits_all.append(self.line_freq)
            self.grid_layout.addWidget(self.line_freq, 6, 20, 1, 4)

            self.line_value = QLineEdit(self)
            self.line_value.setReadOnly(True)
            self.line_value.setText("Różnica amplitudy")
            self.line_edits_all.append(self.line_value)
            self.grid_layout.addWidget(self.line_value, 6, 24, 1, 4)

            mse = numpy.mean(self.fftdiff ** 2)


            top_freqs_c = numpy.where(self.fftdiff[:int(self.csv_standard_length/2)] > mse/4)[0]
            top_freqs = []

            for freq in top_freqs_c:
                if freq == 0:
                    if self.fftdiff[freq] > self.fftdiff[freq+1]:
                        top_freqs.append(freq)
                elif freq == self.csv_standard_length:
                    if self.fftdiff[freq] > self.fftdiff[freq-1]:
                        top_freqs.append(freq)
                else:
                    if self.fftdiff[freq] > self.fftdiff[freq-1] and self.fftdiff[freq] > self.fftdiff[freq+1]:
                        top_freqs.append(freq)
                        print(freq, self.fftdiff[freq], self.fftdiff[freq-1], self.fftdiff[freq+1])

            print(top_freqs_c)
            print(top_freqs)
            top_values = self.fftdiff[:int(self.csv_standard_length/2)][top_freqs]
            
            sorted_pairs = sorted(zip(top_values, top_freqs), reverse=True)

            top_values, top_freqs = zip(*sorted_pairs)

            top_values = list(top_values)
            top_freqs = list(top_freqs)

            row_range = 0

            if len(top_freqs) > 20:
                row_range = 20
            else:
                row_range = len(top_freqs)

            for row in range(row_range):
                for col in range(2):
                    line_edit = QLineEdit(self)
                    line_edit.setReadOnly(True)
                    if col == 0:
                        line_edit.setText(str(top_freqs[row]) + " Hz")
                        self.grid_layout.addWidget(line_edit, row+7, 20, 1, 4)
                    elif col == 1:
                        line_edit.setText(str(round(float(top_values[row]), 3)))
                        self.grid_layout.addWidget(line_edit, row+7, 24, 1, 4)
                    self.line_edits.append(line_edit)
                    self.line_edits_all.append(line_edit)

            self.mse_value = QLineEdit(self)
            self.mse_value.setReadOnly(True)
            self.mse_value.setText("Średni błąd kwadratowy: " + str(round(float(mse), 4)))
            self.line_edits_all.append(self.mse_value)
            self.grid_layout.addWidget(self.mse_value, 28, 20, 1, 8)

            for line_edit in self.line_edits_all:
                line_edit.setStyleSheet("""
                    background-color: #E0E0E0;
                    border: 1px solid #000000;
                    font-size: 14px;
                """)


        elif self.analyze_selector == 0 or self.analyze_selector == 2:
            if hasattr(self, "line_value") and self.line_value is not None:
                self.line_value.setParent(None)
                self.line_value.deleteLater()
                self.line_value = None
                for line_edit in self.line_edits:
                    self.grid_layout.removeWidget(line_edit)
                    line_edit.deleteLater()
                    line_edit = None

            if hasattr(self, "line_freq") and self.line_freq is not None:
                self.line_freq.setParent(None)
                self.line_freq.deleteLater()
                self.line_freq = None

            if hasattr(self, "mse_value") and self.mse_value is not None:
                self.mse_value.setParent(None)
                self.mse_value.deleteLater()
                self.mse_value = None
            

    def analyze_mode_selector(self):
        if self.analyze_selector == 0:
            print("Brak możliwości zmiany trybu analizy")
            new_value = 0
        elif self.analyze_selector == 1:
            self.analyze_mode1()
            self.analyze_mode2()
            self.analyze_mode3()
            self.analyze_mode4()
            new_value = 2
        elif self.analyze_selector == 2:
            self.analyze_mode3()
            self.analyze_mode4()
            self.analyze_mode1()
            self.analyze_mode2()
            new_value = 1
        self.analyze_selector = new_value
        



        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Cockpit()
    main_window.show()
    sys.exit(app.exec())