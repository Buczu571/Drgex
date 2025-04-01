import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QListWidget, QListWidgetItem


class SelectMachineDialog(QDialog):
    def __init__(self, parent=None, machine_list=None):
        super().__init__(parent)
        self.setWindowTitle("Wybór urządzenia")
        self.setGeometry(200, 200, 200, 400)

        self.selected_analyze = []

        self.layout = QVBoxLayout()

        self.machine_list_widget = QListWidget()
        self.machine_list_widget.addItems(machine_list)
        self.layout.addWidget(self.machine_list_widget)

        self.select_button = QPushButton("Wybierz")
        self.select_button.clicked.connect(self.on_confirm)
        self.layout.addWidget(self.select_button)

        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_machine_name(self):
        selected_item = self.machine_list_widget.currentItem()
        return selected_item.text() 


class NewMachineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dodawanie urządzenia")
        self.setGeometry(200, 200, 400, 100)

        self.layout = QVBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Podaj nazwę urządzenia")

        self.confirm_button = QPushButton("Zatwierdź")
        self.confirm_button.clicked.connect(self.on_confirm)

        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_machine_name(self):
        return self.input_field.text().strip()
    

class EditMachineDialog(QDialog):
    def __init__(self, parent=None, selected_machine=None):
        super().__init__(parent)
        self.setWindowTitle("Edycja urządzenia")
        self.setGeometry(200, 200, 400, 100)

        self.layout = QVBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(selected_machine)

        self.confirm_button = QPushButton("Zatwierdź")
        self.confirm_button.clicked.connect(self.on_confirm)

        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_machine_name(self):
        return self.input_field.text().strip()


class SelectMeasurmentsDialog(QDialog):
    def __init__(self, parent=None, measurments_list=None):
        super().__init__(parent)
        self.setWindowTitle("Wybór zestawu pomiarów")
        self.setGeometry(200, 200, 200, 400)

        self.layout = QVBoxLayout()

        self.measurments_list_widget = QListWidget()
        self.measurments_list_widget.addItems(measurments_list)
        self.layout.addWidget(self.measurments_list_widget)

        self.select_button = QPushButton("Wybierz")
        self.select_button.clicked.connect(self.on_confirm)
        self.layout.addWidget(self.select_button)

        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_measurments_name(self):
        selected_item = self.measurments_list_widget.currentItem()
        return selected_item.text() 


class NewMeasurmentsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dodawanie zestawu pomiarów")
        self.setGeometry(200, 200, 400, 100)

        self.layout = QVBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Podaj nazwę zestawu pomiarów")

        self.confirm_button = QPushButton("Zatwierdź")
        self.confirm_button.clicked.connect(self.on_confirm)

        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_measurments_name(self):
        return self.input_field.text().strip()
    

class EditMeasurmentsDialog(QDialog):
    def __init__(self, parent=None, selected_measurments=None):
        super().__init__(parent)
        self.setWindowTitle("Edycja zestawu pomiarów")
        self.setGeometry(200, 200, 400, 100)

        self.layout = QVBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(selected_measurments)

        self.confirm_button = QPushButton("Zatwierdź")
        self.confirm_button.clicked.connect(self.on_confirm)

        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        self.accept()

    def get_measurments_name(self):
        return self.input_field.text().strip()  
    

class SelectAnalyzeDialog(QDialog):
    def __init__(self, parent=None, measurments_list=None):
        super().__init__(parent)
        self.setWindowTitle("Wybór zestawów pomiarów do analizy")
        self.setGeometry(200, 200, 300, 400)

        self.layout = QVBoxLayout()

        self.measurments_list_widget = QListWidget()

        for measurement in measurments_list:
            item = QListWidgetItem(measurement)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.measurments_list_widget.addItem(item)

        self.layout.addWidget(self.measurments_list_widget)

        self.select_button = QPushButton("Wybierz")
        self.select_button.clicked.connect(self.on_confirm)
        self.layout.addWidget(self.select_button)

        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

    def on_confirm(self):
        selected_items = 0
        
        for row in range(self.measurments_list_widget.count()):
            item = self.measurments_list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected_items += 1
        
        if selected_items != 2:
            QMessageBox.warning(self, "Błąd", "Należy wybrać dokładnie 2 zestawy!")
        else:
            self.selected_analyze = []
            for row in range(self.measurments_list_widget.count()):
                item = self.measurments_list_widget.item(row)
                if item.checkState() == Qt.CheckState.Checked:
                    self.selected_analyze.append(item.text())

            self.accept()

    def get_analyze_name(self):
        return self.selected_analyze