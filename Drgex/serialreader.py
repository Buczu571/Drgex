import sys
import serial
import serial.tools.list_ports
import time
import struct
import numpy
import matplotlib.pyplot as plt
import csv
import os
import scipy.signal
from scipy.signal import iirnotch
from scipy.signal import filtfilt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QComboBox, QLabel, QLineEdit, QCheckBox, QVBoxLayout, QSpacerItem, QSizePolicy, QFileDialog
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import random

class SerialReader(QMainWindow):
    def __init__(self, selected_machine, selected_measurment):
        super().__init__()

        self.setWindowTitle("Pomiar")
        self.setGeometry(0, 0, 1536, 864)
        self.showMaximized()
        #self.showFullScreen()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.export_data = []
        self.import_data = []
        self.spectrogram_data = []
        self.spectr_freq_value = None
        self.selected_machine = selected_machine
        self.selected_measurment = selected_measurment

        grid_layout = QGridLayout()

        self.combo_box = QComboBox()
        self.load_com_ports()
        grid_layout.addWidget(self.combo_box, 0, 0, 1, 12)

        time_label = QLabel("Czas pomiaru (s)")
        grid_layout.addWidget(time_label, 0, 12, 1, 2)

        self.time_input = QLineEdit()
        grid_layout.addWidget(self.time_input, 0, 14, 1, 2)

        notch_label = QLabel("Notch Filter")
        grid_layout.addWidget(notch_label, 0, 16, 1, 2)

        self.notch_checkbox = QCheckBox()
        grid_layout.addWidget(self.notch_checkbox, 0, 18, 1, 2)

        self.freq_input = QLineEdit()
        self.freq_input.setPlaceholderText("Freq [Hz]")
        self.freq_input.setText("50")
        grid_layout.addWidget(self.freq_input, 0, 20, 1, 2)

        self.pow_input = QLineEdit()
        self.pow_input.setPlaceholderText("Pow")
        self.pow_input.setText("30")
        grid_layout.addWidget(self.pow_input, 0, 22, 1, 2)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_measurement)
        grid_layout.addWidget(self.start_button, 0, 24, 1, 8)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.stop_button)


        for row in range(32):
            for col in range(32):
                window_width = self.width()
                window_height = self.height()

                spacer_width = int(window_width / 32)
                spacer_height = int(window_height / 32)

                spacer = QSpacerItem(spacer_width, spacer_height, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                
                grid_layout.addItem(spacer, row, col)

        self.plot_values_widget = QWidget(self)
        self.plot_values_layout = QVBoxLayout(self.plot_values_widget)

        self.plot_values_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        self.values_toolbar = NavigationToolbar2QT(self.plot_values_canvas, self)
        self.plot_values_layout.addWidget(self.values_toolbar)

        self.plot_values_layout.addWidget(self.plot_values_canvas)

        grid_layout.addWidget(self.plot_values_widget, 2, 0, 30, 16)

        self.plot_fft_widget = QWidget(self)
        self.plot_fft_layout = QVBoxLayout(self.plot_fft_widget)

        self.plot_fft_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        self.fft_toolbar = NavigationToolbar2QT(self.plot_fft_canvas, self)
        self.plot_fft_layout.addWidget(self.fft_toolbar)

        self.plot_fft_layout.addWidget(self.plot_fft_canvas)

        grid_layout.addWidget(self.plot_fft_widget, 2, 16, 30, 16)

        self.text_field_1 = QLineEdit(self)
        self.text_field_2 = QLineEdit(self)
        self.text_field_3 = QLineEdit(self)
        self.text_field_4 = QLineEdit(self)

        self.text_field_1.setReadOnly(True)
        self.text_field_2.setReadOnly(True)
        self.text_field_3.setReadOnly(True)
        self.text_field_4.setReadOnly(True)

        self.text_field_1.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        self.text_field_2.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        self.text_field_3.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        self.text_field_4.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        grid_layout.addWidget(self.text_field_1, 1, 0, 1, 2)
        grid_layout.addWidget(self.text_field_2, 1, 2, 1, 2)
        grid_layout.addWidget(self.text_field_3, 1, 4, 1, 2)
        grid_layout.addWidget(self.text_field_4, 1, 6, 1, 2)

        self.csv_import_button = QPushButton("Importuj CSV")
        self.csv_import_button.clicked.connect(self.import_csv)
        grid_layout.addWidget(self.csv_import_button, 1, 24, 1, 4)

        self.csv_export_button = QPushButton("Eksportuj CSV")
        self.csv_export_button.clicked.connect(self.export_csv)
        grid_layout.addWidget(self.csv_export_button, 1, 28, 1, 4)

        self.spectrogram_button = QPushButton("Spektrogram")
        self.spectrogram_button.clicked.connect(self.spectrogram)
        grid_layout.addWidget(self.spectrogram_button, 1, 8, 1, 4)


        samplesno_label = QLabel("Wiele próbek")
        grid_layout.addWidget(samplesno_label, 1, 16, 1, 2)

        self.samplesno_checkbox = QCheckBox()
        grid_layout.addWidget(self.samplesno_checkbox, 1, 18, 1, 2)

        self.samplesno_input = QLineEdit()
        self.samplesno_input.setPlaceholderText("Ilość próbek")
        grid_layout.addWidget(self.samplesno_input, 1, 20, 1, 4)

        central_widget.setLayout(grid_layout)

    def load_com_ports(self):
        try:
            ports = serial.tools.list_ports.comports()
            for port, desc, hwid in sorted(ports):
                port_data = ("{}: {}".format(port, desc))
                self.combo_box.addItem(port_data)  
        except Exception as e:
            print(f"Nie udało się pobrać portów COM: {e}")

    def start_measurement(self):
        try:
            used_port = self.combo_box.currentText()
            used_port = used_port.split(":")[0]
        except:
            print("Błąd: Nieprawidłowy port COM!")
            return

        try:
            time_value = int(self.time_input.text())
            if time_value <= 0:
                print("Błąd: Czas pomiaru musi być liczbą większą od zera!")
                return
            self.start_button.setEnabled(False)
            self.timer.start(time_value * 1000)
        except ValueError:
            print("Błąd: Czas pomiaru musi być liczbą całkowitą!")
            return
        
        #print(f"Wybrany port COM: {used_port}")
        #print(f"Czas pomiaru: {time_value} sekundy")

        self.plot_values_canvas.figure.clf()
        self.plot_fft_canvas.figure.clf()

        PORT = used_port
        BAUDRATE = 1500000
        TIMEOUT = 1

        ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)

        if self.samplesno_checkbox.isChecked():
            try:
                samplesno_value = int(self.samplesno_input.text())
                if samplesno_value <= 0:
                    print("Błąd: Liczba próbek musi być liczbą większą od zera!")
                    return
            except ValueError:
                print("Błąd: Liczba próbek musi być liczbą całkowitą!")
                return
            
            if self.selected_machine == None or self.selected_measurment == None:
                print("Błąd: Brak wybranej maszyny lub zestawu danych")
                return
            
            for i in range(0, samplesno_value):
                self.text_field_4.setText(str(i+1))
                start_time = time.time()
                end_time = time_value
                data = []
                err_no = 0
                
                while time.time() - start_time < end_time:
                    raw_data = ser.read(2)
                    if len(raw_data) == 2:
                        adc_value = struct.unpack('<h', raw_data)[0]
                        if adc_value < 0 or adc_value > 4095:
                            print("ADC Value Error", adc_value)
                            err_no = err_no + 1
                            ser.close()
                            time.sleep(0.1)
                            ser.open()
                            start_time = time.time()
                            data = []
                        else:
                            data.append(adc_value)

                print("Zakończono odbieranie danych.", len(data), err_no)

                self.export_data = data.copy()
                self.export_data.insert(0,int(len(self.export_data)/time_value))

                self.spectrogram_data = data.copy()
                self.spectrogram_data.insert(0,int(len(self.export_data)/time_value))

                data = numpy.array(data, dtype=float)

                if self.notch_checkbox.isChecked():
                    try:
                        freq_value = int(self.freq_input.text())
                        if freq_value <= 0:
                            print("Błąd: Częstotliwość filtru musi być liczbą większą od zera!")
                            return
                    except ValueError:
                        print("Błąd: Częstotliwość filtru musi być liczbą całkowitą!")
                        return
                    
                    try:
                        pow_value = int(self.pow_input.text())
                        if pow_value <= 0:
                            print("Błąd: Moc tłumienia filtru musi być liczbą większą od zera!")
                            return
                    except ValueError:
                        print("Błąd: Moc tłumienia filtru musi być liczbą całkowitą!")
                        return
                    
                    data = self.notch_filter(data, freq_value, int(len(data)/time_value), pow_value)

                base_dir = os.path.dirname(os.path.abspath(__file__))
                file_name = os.path.join(base_dir, self.selected_machine, self.selected_measurment, str(str(i+1)+".csv"))

                if file_name:
                    with open(file_name, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        for value in self.export_data:
                            writer.writerow([value])
                    print(f"Plik zapisany jako {file_name}")

        else:
            start_time = time.time()
            end_time = time_value
            data = []
            err_no = 0
            
            while time.time() - start_time < end_time:
                self.text_field_4.setText(str(end_time - start_time))
                raw_data = ser.read(2)
                if len(raw_data) == 2:
                    adc_value = struct.unpack('<h', raw_data)[0]
                    if adc_value < 0 or adc_value > 4095:
                        print("ADC Value Error", adc_value)
                        err_no = err_no + 1
                        ser.close()
                        time.sleep(0.1)
                        ser.open()
                        start_time = time.time()
                        data = []
                    else:
                        data.append(adc_value)

            print("Zakończono odbieranie danych.", len(data), err_no)
            self.text_field_4.setText("")

            self.export_data = data.copy()
            self.export_data.insert(0,int(len(self.export_data)/time_value))

            self.spectrogram_data = data.copy()
            self.spectrogram_data.insert(0,int(len(self.export_data)/time_value))

            data = numpy.array(data, dtype=float)
            self.plot_values_data(data)

            if self.notch_checkbox.isChecked():
                try:
                    freq_value = int(self.freq_input.text())
                    if freq_value <= 0:
                        print("Błąd: Częstotliwość filtru musi być liczbą większą od zera!")
                        return
                except ValueError:
                    print("Błąd: Częstotliwość filtru musi być liczbą całkowitą!")
                    return
                
                try:
                    pow_value = int(self.pow_input.text())
                    if pow_value <= 0:
                        print("Błąd: Moc tłumienia filtru musi być liczbą większą od zera!")
                        return
                except ValueError:
                    print("Błąd: Moc tłumienia filtru musi być liczbą całkowitą!")
                    return
                
                data = self.notch_filter(data, freq_value, int(len(data)/time_value), pow_value)

            data_fft = list(map(int, data))
            print(numpy.mean(data_fft))

            self.text_field_1.setText(str(len(data)))
            self.text_field_2.setText(str(int(len(data)/time_value)))
            self.text_field_3.setText(f"{float(numpy.mean(data_fft)):.2f}")

            data_fft = [(i / 4096) * 3.3 for i in data_fft]

            data_fft = numpy.array(data_fft) - numpy.mean(data_fft)

            yf = numpy.fft.fft(data_fft)
            yf = numpy.abs(yf) * 2048 / len(data_fft)
            xf = numpy.fft.fftfreq(len(data_fft), 1/(len(data_fft)/time_value))

            print(len(data_fft),len(data_fft)/time_value)

            self.plot_fft_data(data_fft, yf, xf)

    def stop_button(self):
        #MT
        self.start_button.setEnabled(True)
        self.timer.stop()

    def plot_values_data(self, data):
        ax = self.plot_values_canvas.figure.add_subplot(111)

        ax.clear()
        ax.plot(data, marker='o', linestyle='-')

        ax.set_xlabel("Numer próbki")
        ax.set_ylabel("Wartość ADC")
        ax.set_title("Wykres wartości ADC")

        self.plot_values_canvas.draw()

    def plot_fft_data(self, data_fft, yf, xf):
        ax = self.plot_fft_canvas.figure.add_subplot(111)

        ax.clear()
        ax.plot(xf[:len(data_fft)//2], numpy.abs(yf[:len(data_fft)//2]))

        ax.set_xlabel("Częstotliwość (Hz)")
        ax.set_ylabel("Amplituda")
        ax.set_title("Analiza FFT")

        self.plot_fft_canvas.draw()

    def notch_filter(self, data, freq, fs, quality):
        b, a = iirnotch(freq, quality, fs)
        return filtfilt(b, a, data)
    

    def spectrogram(self):
        try:
            first_element = self.spectrogram_data[0]
        except IndexError:
            print("Brak danych do spektrogramu")
            return

        if isinstance (self.spectrogram_data, list):
            self.spectr_freq_value = self.spectrogram_data[0]
            self.spectrogram_data.pop(0)

        self.spectrogram_data = numpy.array(self.spectrogram_data)

        frequencies, times, Sxx = scipy.signal.spectrogram(self.spectrogram_data, self.spectr_freq_value, nperseg=1024)

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, 10 * numpy.log10(Sxx), shading='gouraud', cmap='inferno')
        plt.colorbar(label="Moc sygnału")
        plt.xlabel("Czas (s)")
        plt.ylabel("Częstotliwość (Hz)")
        plt.title("Spektrogram sygnału")
        plt.ylim(0, self.spectr_freq_value/2)
        plt.show()


    def import_csv(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if self.selected_machine and self.selected_measurment != None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik CSV", os.path.join(base_dir, self.selected_machine, self.selected_measurment), "CSV Files (*.csv);;All Files (*)")
        elif self.selected_machine != None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik CSV", os.path.join(base_dir, self.selected_machine), "CSV Files (*.csv);;All Files (*)")
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik CSV", "", "CSV Files (*.csv);;All Files (*)")

        if file_name:
            with open(file_name, mode="r", newline="") as file:
                reader = csv.reader(file)
                self.import_data = [int(row[0]) for row in reader]

            print(f"Dane zaimportowane z pliku {file_name}")

            self.spectrogram_data = self.import_data.copy()

            self.plot_values_canvas.figure.clf()
            self.plot_fft_canvas.figure.clf()

            freq_value = self.import_data[0]
            self.import_data.pop(0)

            data = numpy.array(self.import_data, dtype=float)
            self.plot_values_data(data)

            data_fft = list(map(int, data))

            self.text_field_1.setText(str(len(data)))
            self.text_field_2.setText(str(int(freq_value)))
            self.text_field_3.setText(f"{float(numpy.mean(data_fft)):.2f}")

            data_fft = [(i / 4096) * 3.3 for i in data_fft]

            data_fft = numpy.array(data_fft) - numpy.mean(data_fft)

            yf = numpy.fft.fft(data_fft)
            yf = numpy.abs(yf) * 2048 / len(data_fft)
            xf = numpy.fft.fftfreq(len(data_fft), 1/(freq_value))

            self.plot_fft_data(data_fft, yf, xf)


    def export_csv(self):

        base_dir = os.path.dirname(os.path.abspath(__file__))

        if self.selected_machine and self.selected_measurment != None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz plik CSV", os.path.join(base_dir, self.selected_machine, self.selected_measurment), "CSV Files (*.csv);;All Files (*)")
        elif self.selected_machine != None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz plik CSV", os.path.join(base_dir, self.selected_machine), "CSV Files (*.csv);;All Files (*)")
        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz plik CSV", "", "CSV Files (*.csv);;All Files (*)")

        if file_name:
            with open(file_name, mode="w", newline="") as file:
                writer = csv.writer(file)
                for value in self.export_data:
                    writer.writerow([value])
            print(f"Plik zapisany jako {file_name}")


#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    window = SerialReader()
#    window.show()
#    sys.exit(app.exec())