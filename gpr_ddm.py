import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lhs_sampling(ranges, num_samples):
    """
    generate Latin Hypercube Samples within given ranges
    
    parameters:
    ranges: list of tuples [(min1, max1), (min2, max2), ...] for each dimension
    
            tuples maintain order of the data inserted
            tuples are immutable (Cant be changed after creation)
            different types of data can be stored in a tuple (Tho we dont care here)
            and allow duplicates
            example: [(0, 1), (10, 20)] for 2D sampling
    
    num_samples: number of samples to generate (Remember f tilde, not f, depending on how many samples you have, thats your prediction)
    
    returns:
    numpy array of shape (num_samples, n_dimensions)
            such that each row represents a sample point in the parameter space
            example: For 2D sampling with 5 samples, shape will be (5, 2)

    """
    n_dimensions = len(ranges)
    print(f"generating {num_samples} LHS in {n_dimensions} dimensions")
    
    #each row is a sample, each column is a dimension (2D for this case)
    samples = np.zeros((num_samples, n_dimensions))
    
    for dim in range(n_dimensions):
        """
        generate LHS samples for this dimension
        so the intervals are divided into num_samples parts, for the normalized [0,1] 
        range we have num_samples intervals (+1 because we start at 0 btw)
        
        so if n=5 and range is 0-1, intervals are:
        0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        """
        intervals = np.linspace(0, 1, num_samples + 1)
        
        """
        this generates random points within interval created above
        intervals[:-1] = [0.0, 0.2, 0.4, 0.6, 0.8] 
        intervals[1:] = [0.2, 0.4, 0.6, 0.8, 1.0]

        basically between 0.0-0.2, 0.2-0.4, etc
        5 samples will be generated, one in each interval
        
        sm like this:
        |-------|-------|-------|-------|-------|
        0.0    0.2    0.4    0.6    0.8    1.0
            ↓       ↓       ↓       ↓       ↓
            0.13   0.37   0.45    0.71    0.92
        """
        lhs_samples = np.random.uniform(intervals[:-1], intervals[1:])
        
        """
        shuffle to avoid any correlation between dimensions
        meaning each dimension is treated independently
        beacause if we dont shuffle, we get the same pattern such as:
        B ↑
        |     ●
        |   ●
        | ●
        |●
        |●________→ A
        and values fall on a diagonal line, which is not desired
        
        so after shuffling we get sm like:
        B ↑
        |   ●   ●
        |     
        | ●   ●
        |   
        |●________→ A
        """
        np.random.shuffle(lhs_samples)
        
        """
        since we had it normalized between [0, 1], scale it to the given range
        so for example: range (10, 20) and a sample of 0.37, we get:
        10 + 0.37 * (20 - 10) = 13.7
        """
        min_val, max_val = ranges[dim] #range for dimension 1 and for dimension 2
        samples[:, dim] = min_val + lhs_samples * (max_val - min_val) #all rows (samples), but only column (dimension)
    
    return samples

def plot_and_display_samples(samples, ranges, title="Latin Hypercube Sampling (LHS) for Airfoil Parameters"):
    """
    Plot samples and display them in a table format.
    
    Parameters:
    samples: numpy array of samples
    ranges: list of tuples for axis limits
    title: plot title
    """
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(samples[:, 0], samples[:, 1], color='purple', s=80, alpha=0.7, edgecolors='#6b2c91') #uni yellow #e2b525
    plt.xlim(ranges[0])
    plt.ylim(ranges[1])
    plt.xlabel('p (Location of Max Camber)') #q1 #p belongs [1.5,5]
    plt.ylabel('t (Airfoil Half Thickness)') #q2 #t belongs [12,25]
    #m is kept constant at 4
    plt.title(title)
    plt.grid(True, alpha=0.3) #alpha is transparency btw
    
    #this adds sample numbers as annotations (not necessary but helpful)
    for i, (x, y) in enumerate(samples):
        plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Subplot 2: Table (Showing the samples in tabular form but we gonna use latex for it anyways)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Create table data
    table_data = []
    for i, sample in enumerate(samples):
        table_data.append([f"Sample {i+1}", f"{sample[0]:.4f}", f"{sample[1]:.4f}"])
    
    # Create table
    table = plt.table(cellText=table_data,
                     colLabels=['Sample #', 'X Value', 'Y Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#e2b525')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    plt.show()
    
    # Also print the table to console
    df = pd.DataFrame(samples, columns=['X Parameter', 'Y Parameter'])
    df.index = [f'Sample {i+1}' for i in range(len(samples))]
    print("\nLatin Hypercube Samples:")
    print(df.round(4))


#user parameters
x_range = (1.5, 5.0)  #q1 #p belongs [1.5,5]
y_range = (12, 25)  #q2 #t belongs [12,25]
num_samples = 20

#usage
ranges = [x_range, y_range]
lhs_samples = lhs_sampling(ranges, num_samples)

plot_and_display_samples(lhs_samples, ranges)