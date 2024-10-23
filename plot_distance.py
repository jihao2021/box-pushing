import csv
import numpy as np
import matplotlib.pyplot as plt
import constant

def plot_distance(file_path):

    NUMBER_OF_AGENT = count_rows_csv(file_path)
    distance_list = [[] for _ in range(NUMBER_OF_AGENT)]
    color_list = constant.color_list

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        combined_distance = next(readCSV)  # Read the first row

        # Convert the list of strings to a list of integers
        combined_distance = [int(value) for value in combined_distance]

        print(len(combined_distance))

        plt.plot(combined_distance, color='blue', label='Smoothed Average Distance')

        plt.legend()
        plt.ylabel('Distance')
        plt.xlabel('Training Episode (every 100 episodes)')
        plt.show()

def count_rows_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
        return row_count
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

file_path = './training_output/distance_output/distance30.csv'
plot_distance(file_path)