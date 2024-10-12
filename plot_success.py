import csv
import numpy as np
import matplotlib.pyplot as plt
import constant  # Ensure this module exists and color_list is defined.

def plot_success(file):
    file_path = f"./training_output/success_output/success{file}"

    # Read success data from the CSV file
    with open(file_path, newline='') as csvfile:
        # Read the first line and split by commas
        row = csvfile.readline().strip().split(',')
        success_list = [int(value) for value in row]  # Convert to integers

    # Calculate the average of every 25 entries
    chunk_size = 25

    print(len(success_list))

    averages = [np.mean(success_list[i:i + chunk_size]) for i in range(0, len(success_list), chunk_size)]

    # Plot the averages
    plt.plot(averages, color='blue')  # Use the first color in the list
    plt.title('Average Success Probability')
    plt.xlabel('Chunk Index (25 entries per chunk)')
    plt.ylabel('Average Success Probability')
    plt.xticks(ticks=range(len(averages)), labels=[str(i * chunk_size) for i in range(len(averages))])  # X-axis labels
    plt.grid()
    plt.show()

# Example usage
file = '15.csv'
plot_success(file)
