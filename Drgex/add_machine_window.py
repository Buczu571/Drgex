from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
import os

class AddMachineWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Dodaj Maszynę")
        self.setGeometry(100, 100, 300, 200)

        self.layout = QVBoxLayout(self)
        self.machine_name_input = QLineEdit(self)
        self.machine_name_input.setPlaceholderText("Wpisz nazwę maszyny")
        self.technical_data_input = QLineEdit(self)
        self.technical_data_input.setPlaceholderText("Wpisz dane techniczne")
        self.location_input = QLineEdit(self)
        self.location_input.setPlaceholderText("Wpisz lokalizację")
        self.additional_info_input = QLineEdit(self)
        self.additional_info_input.setPlaceholderText("Wpisz dodatkowe informacje")
        self.save_button = QPushButton("Zapisz", self)
        self.save_button.clicked.connect(self.save_machine)

        self.layout.addWidget(self.machine_name_input)
        self.layout.addWidget(self.technical_data_input)
        self.layout.addWidget(self.location_input)
        self.layout.addWidget(self.additional_info_input)
        self.layout.addWidget(self.save_button)

    def save_machine(self):
        machine_name = self.machine_name_input.text()
        technical_data = self.technical_data_input.text()
        location = self.location_input.text()
        additional_info = self.additional_info_input.text()

        self.main_window.add_machine(machine_name, technical_data, location, additional_info)
        self.save_to_file(machine_name, technical_data, location, additional_info)
        self.close()
        
    def save_to_file(self, machine_name, technical_data, location, additional_info):
        folder_name = "machines"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        filename = os.path.join(folder_name, f"{machine_name}.txt")
        if os.path.exists(filename):
            QMessageBox.warning(self, "Błąd", f"Plik o nazwie '{filename}' już istnieje. Wybierz inną nazwę.")
            return
        
        with open(filename, 'w') as file:
            file.write(f"Nazwa maszyny: {machine_name}\n")
            file.write(f"Dane techniczne: {technical_data}\n")
            file.write(f"Lokalizacja: {location}\n")
            file.write(f"Dodatkowe informacje: {additional_info}\n")
        
        QMessageBox.information(self, "Sukces", f"Dane maszyny '{machine_name}' zostały zapisane.")