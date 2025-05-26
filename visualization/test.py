import matplotlib.pyplot as plt
import numpy as np

def maxPlot(data, label='Data'):
    """
    Function to plot the maximum values of a dataset.
    
    Parameters:
    - data: List or array of numerical data to be plotted
    - label: Label for the dataset in the plot
    """
    # Calculate the maximum value and its index
    max_value = np.max(data)
    max_index = np.argmax(data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=label)
    plt.scatter(max_index, max_value, color='red', zorder=5)  # Highlight max value
    plt.text(max_index, max_value, f'Max: {max_value}', fontsize=12, ha='right', color='red')
    plt.title('Plot with Maximum Value Highlighted')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('max_plot.png')  # Save the plot as a PNG file

# Example usage
data = np.random.random(100)  # Random data for demonstration
maxPlot(data)
