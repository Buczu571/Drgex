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
import tensorflow as tf
from keras.api.models import *
from keras.api.layers import *
from scipy.signal import iirnotch
from scipy.signal import filtfilt
from scipy.fft import fft, fftfreq
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QComboBox, QLabel, QLineEdit, QCheckBox, QVBoxLayout, QSpacerItem, QSizePolicy, QFileDialog
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure


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

        grid_layout.addWidget(self.plot_values_widget, 2, 4, 30, 14)

        self.plot_fft_widget = QWidget(self)
        self.plot_fft_layout = QVBoxLayout(self.plot_fft_widget)

        self.plot_fft_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        self.fft_toolbar = NavigationToolbar2QT(self.plot_fft_canvas, self)
        self.plot_fft_layout.addWidget(self.fft_toolbar)

        self.plot_fft_layout.addWidget(self.plot_fft_canvas)

        grid_layout.addWidget(self.plot_fft_widget, 2, 18, 30, 14)

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

        prediction_label = QLabel("Obliczanie prawdopodobieństwa nieprawidłowości")
        grid_layout.addWidget(prediction_label, 2, 0, 1, 3)

        self.preditction_checkbox = QCheckBox()
        grid_layout.addWidget(self.preditction_checkbox, 2, 3, 1, 1)

        expectedrpm_label = QLabel("Zakładane RPM")
        grid_layout.addWidget(expectedrpm_label, 3, 0, 1, 2)

        self.expectedrpm_input = QLineEdit()
        self.expectedrpm_input.setPlaceholderText("RPM")
        grid_layout.addWidget(self.expectedrpm_input, 3, 2, 1, 2)

        pred_imbalance_label = QLabel("Niewyważenie [%]")
        grid_layout.addWidget(pred_imbalance_label, 4, 0, 1, 2)

        self.imblanace_value = QLineEdit(self)
        self.imblanace_value.setReadOnly(True)
        grid_layout.addWidget(self.imblanace_value, 4, 2, 1, 2)

        self.imblanace_value.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        pred_imbalance_pul_label = QLabel("Prędkość drgań [mm/s]")
        grid_layout.addWidget(pred_imbalance_pul_label, 5, 0, 1, 2)

        self.imblanace_pul_value = QLineEdit(self)
        self.imblanace_pul_value.setReadOnly(True)
        grid_layout.addWidget(self.imblanace_pul_value, 5, 2, 1, 2)

        self.imblanace_pul_value.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        pred_imbalance_pul_freq_label = QLabel("Częstotliwość pracy maszyny [Hz]")
        grid_layout.addWidget(pred_imbalance_pul_freq_label, 6, 0, 1, 2)

        self.imblanace_pul_freq = QLineEdit(self)
        self.imblanace_pul_freq.setReadOnly(True)
        grid_layout.addWidget(self.imblanace_pul_freq, 6, 2, 1, 2)

        self.imblanace_pul_freq.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

        prediction_hardness_label = QLabel("Obliczanie twardości materiału")
        grid_layout.addWidget(prediction_hardness_label, 8, 0, 1, 3)

        self.preditction_hardness_checkbox = QCheckBox()
        grid_layout.addWidget(self.preditction_hardness_checkbox, 8, 3, 1, 1)

        pred_hardness_label = QLabel("Twardość [GPa]")
        grid_layout.addWidget(pred_hardness_label, 9, 0, 1, 2)

        self.hardness_value = QLineEdit(self)
        self.hardness_value.setReadOnly(True)
        grid_layout.addWidget(self.hardness_value, 9, 2, 1, 2)

        self.hardness_value.setStyleSheet("""
            background-color: #E0E0E0;
            border: 1px solid #000000;
        """)

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

            if self.preditction_checkbox.isChecked():
                try:
                    expectedrpm_input = int(self.expectedrpm_input.text())
                    if expectedrpm_input <= 0:
                        print("Błąd: Zakładane RPM musi być liczbą większą od zera!")
                        return
                except ValueError:
                    print("Błąd: Zakładane RPM musi być liczbą całkowitą!")
                    return
                
                self.predict(data, expectedrpm_input)
                self.predict_pul(data, expectedrpm_input)

            if self.preditction_hardness_checkbox.isChecked():
                self.hardness_predict(data)

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

    def gaussian_filter(self, freq, center_freq, sigma):
        return numpy.exp(-0.5 * ((freq - center_freq) / sigma) ** 2)
    

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

            if self.preditction_checkbox.isChecked():
                try:
                    expectedrpm_input = int(self.expectedrpm_input.text())
                    if expectedrpm_input <= 0:
                        print("Błąd: Zakładane RPM musi być liczbą większą od zera!")
                        return
                except ValueError:
                    print("Błąd: Zakładane RPM musi być liczbą całkowitą!")
                    return
                
                self.predict(data, expectedrpm_input)
                self.predict_pul(data, expectedrpm_input)

            if self.preditction_hardness_checkbox.isChecked():            
                self.hardness_predict(data)

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



    def predicion_color_change(self, line_edit: QLineEdit, percent_value: float):
        if percent_value < 0 or percent_value > 100:
            raise ValueError("Błąd wartości")

        if percent_value <= 20:
            line_edit.setStyleSheet("""
            background-color: #D0F0C0;
            """)
        elif percent_value <= 40:
            line_edit.setStyleSheet("""
            background-color: #F0F8A0;
            """)
        elif percent_value <= 60:
            line_edit.setStyleSheet("""
            background-color: #FFF4A3;
            """)
        elif percent_value <= 80:
            line_edit.setStyleSheet("""
            background-color: #FFD580;
            """)
        else:
            line_edit.setStyleSheet("""
            background-color: #FFB3B3;
            """)


    def predict(self, data, expected_rpm):
        SAMPLING_RATE = 24000
        SAMPLE_SEC = len(data)//SAMPLING_RATE

        model = tf.keras.models.load_model('model_cnn_imb1.h5')
        model2 = tf.keras.models.load_model('model_cnn_imb2.h5')

        windows_test = []

        current_window = []
        row_counter = 0

        for row in data:
            if row_counter > SAMPLE_SEC * SAMPLING_RATE:
                break
            current_window.append(row)
            if len(current_window) >= SAMPLING_RATE:
                windows_test.append(current_window)
                current_window = []
            row_counter = row_counter + 1


        fft_test_imb_1 = []
        fft_test_imb_2 = []

        for window in windows_test:
            for i in range(50, 1000, 50):
                window = self.notch_filter(window, i, 24000, 12)

            window = window - numpy.mean(window)

            fft_result = numpy.fft.fft(window)
            
            fft_freq = numpy.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

            fft_magnitude = numpy.abs(fft_result) / (SAMPLING_RATE)
            
            half_n = SAMPLING_RATE // 2
            fft_freq = fft_freq[:half_n]
            fft_magnitude = fft_magnitude[:half_n]
            fft_magnitude = fft_magnitude[:1000]
            
            fft_test_imb_1.append(fft_magnitude)


        for window in windows_test:
            for i in range(50, 4000, 50):
                window = self.notch_filter(window, i, 24000, 24)

            window = window - numpy.mean(window)

            fft_result = numpy.fft.fft(window)
            
            fft_freq = numpy.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

            fft_magnitude = numpy.abs(fft_result) / (SAMPLING_RATE)
            
            expected_freq = expected_rpm / 60

            window_width_hz = 16
            sigma = window_width_hz / 6

            gaussian_window = numpy.exp(-0.5 * ((fft_freq - expected_freq) / sigma) ** 2)

            half_n = 4000

            positive_freq = fft_freq[:half_n]
            positive_fft = fft_magnitude[:half_n]
            gaussian_window_positive = gaussian_window[:half_n]

            weighted_fft = numpy.abs(positive_fft) * gaussian_window_positive

            peak_idx = numpy.argmax(weighted_fft)
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

            gaussian_f1 = self.gaussian_filter(numpy.array(positive_freqs), f1, sigma)
            gaussian_f2 = self.gaussian_filter(numpy.array(positive_freqs), f2, sigma)
            gaussian_f3 = self.gaussian_filter(numpy.array(positive_freqs), f3, sigma)

            filtered_power_f1 = numpy.array(positive_power) * gaussian_f1
            filtered_power_f2 = numpy.array(positive_power) * gaussian_f2
            filtered_power_f3 = numpy.array(positive_power) * gaussian_f3

            filtered_power = filtered_power_f1 + filtered_power_f2 + filtered_power_f3
            filtered_freq = numpy.array(positive_freqs)
            
            fft_test_imb_2.append(filtered_power)

        predictions_imb_model1 = []
        predictions_imb_model2 = []

        X_test1 = numpy.array(fft_test_imb_1)
        X_test1 = X_test1[..., numpy.newaxis]
        X_test2 = numpy.array(fft_test_imb_2)
        X_test2 = X_test2[..., numpy.newaxis]

        predictions1 = model.predict(X_test1)
        predictions2 = model2.predict(X_test2)

        predictions_percent1 = predictions1.flatten() * 100  
        predictions_percent2 = predictions2.flatten() * 100

        for i, prob in enumerate(predictions_percent1):
            predictions_imb_model1.append(prob)

        for i, prob in enumerate(predictions_percent2):
            predictions_imb_model2.append(prob)

        imb_final_result = numpy.mean(predictions_imb_model1 + predictions_imb_model2)

        self.imblanace_value.setText(f"{imb_final_result:.2f}%")
        self.predicion_color_change(self.imblanace_value, imb_final_result)

    
    def predict_pul(self, data, expected_rpm):
        SAMPLING_RATE = 24000

        n = len(data)
        window = numpy.hanning(n)
        yf = fft(data * window)
        xf = fftfreq(n, 1/SAMPLING_RATE)[1:n//2]
        yf = (2.0 / numpy.sum(window)) * numpy.abs(yf[1:n//2])
        velocity_spectrum = (yf / (2 * numpy.pi * xf)) * 1000 

        est_freq = expected_rpm / 60.0
        search_bandwidth = 0.1 * est_freq
        mask = (xf > est_freq - search_bandwidth) & (xf < est_freq + search_bandwidth)
        if not numpy.any(mask):
            print("Brak danych w zakresie 1x RPM.")
            return None, None
        
        freq_range = xf[mask]
        spec_range = velocity_spectrum[mask]

        idx_max = numpy.argmax(spec_range)

        if 0 < idx_max < len(spec_range)-1:
            y0, y1, y2 = spec_range[idx_max-1], spec_range[idx_max], spec_range[idx_max+1]
            x0, x1, x2 = freq_range[idx_max-1], freq_range[idx_max], freq_range[idx_max+1]

            denominator = (y0 - 2*y1 + y2)
            if denominator == 0:
                peak_freq = x1
                peak_value = y1
            else:
                delta = 0.5 * (y0 - y2) / denominator
                peak_freq = x1 + delta * (x2 - x0) / 2
                peak_value = y1 - 0.25 * (y0 - y2) * delta
        else:
            peak_freq = freq_range[idx_max]
            peak_value = spec_range[idx_max]

        self.imblanace_pul_value.setText(f"{peak_value:.2f}")
        self.imblanace_pul_freq.setText(f"{peak_freq:.2f}")


    def hardness_predict(self, data):
        SAMPLING_RATE = 24000
        SAMPLE_SEC = len(data)//SAMPLING_RATE

        model = tf.keras.models.load_model('model_hardness.keras')

        windows_test = []

        current_window = []
        row_counter = 0

        for row in data:
            if row_counter > SAMPLE_SEC * SAMPLING_RATE:
                break
            current_window.append(row)
            if len(current_window) >= SAMPLING_RATE:
                windows_test.append(current_window)
                current_window = []
            row_counter = row_counter + 1

        fft_test = []

        for window in windows_test:
            window = self.notch_filter(window, 50, 24000, 72)

            for i in range(50, 12050, 50):
                window = self.notch_filter(window, i, 24000, 48)

            window = window - numpy.mean(window)

            fft_result = numpy.fft.fft(window)

            fft_freq = numpy.fft.fftfreq(len(window), 1 / SAMPLING_RATE)

            fft_magnitude = numpy.abs(fft_result) / (SAMPLING_RATE)

            half_n = SAMPLING_RATE // 2
            fft_freq = fft_freq[:half_n]
            fft_magnitude = fft_magnitude[:half_n]

            fft_freq = fft_freq[:4000]
            fft_magnitude = fft_magnitude[:4000]

            fft_test.append(fft_magnitude)


        X_test = numpy.array(fft_test)
        X_test = X_test[..., numpy.newaxis]
        predictions = model.predict(X_test)
        predictions = numpy.array(predictions).flatten()

        hardness_final_result = numpy.mean(predictions)

        self.hardness_value.setText(f"{hardness_final_result:.4f}")


#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    window = SerialReader()
#    window.show()
#    sys.exit(app.exec())