import os
from PyQt6.QtWidgets import QMessageBox

def load_machines_from_folder(folder_name="machines"):
    machines = []
    if not os.path.exists(folder_name):
        print(f"Folder '{folder_name}' nie istnieje.")
        return machines
    
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
                machines.append(machine_data)
    return machines

def delete_machine(machine_index, machines, folder_name="machines"):
    machine_name = machines[machine_index]['name']
    del machines[machine_index]
    filename = os.path.join(folder_name, f"{machine_name}.txt")
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Usunięto plik: {filename}")
    else:
        print(f"Plik {filename} nie istnieje.")
    QMessageBox.information(None, "Sukces", f"Maszyna '{machine_name}' została usunięta.")