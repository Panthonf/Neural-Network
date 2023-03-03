import pandas as pd
import numpy as np

import csv

# Set the file path and open the file in read mode
file_path = r'C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\data_preprocessing.txt'
with open(file_path, "r") as csv_file:

    # Create a CSV reader object
    reader = csv.reader(csv_file)

    # Create a CSV writer object for the new file
    new_file_path = r"C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\test1.txt"
    with open(new_file_path, "w", newline='') as new_csv_file:
        writer = csv.writer(new_csv_file)

        # Skip the rows before index 20
        for _ in range(243):
            next(reader)

        # Write the rows between index 20 and 50
        for _ in range(34):
            writer.writerow(next(reader))


# Set the file path and open the file in read mode
file_path = r'C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\data_preprocessing.txt'
with open(file_path, "r") as csv_file:

    # Create a CSV reader object
    reader = csv.reader(csv_file)

    # Create a CSV writer object for the new file
    new_file_path = r"C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\train1.txt"
    with open(new_file_path, "w", newline='') as new_csv_file:
        writer = csv.writer(new_csv_file)

        # Skip the rows before index 20
        for _ in range(1):
            next(reader)

        # Write the rows between index 20 and 50
        for _ in range(243):
            writer.writerow(next(reader))
