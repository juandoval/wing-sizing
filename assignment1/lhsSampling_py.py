import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lhsSampling_py(q1min, q1max, q2min, q2max, n):
    """
    LHS sampling for q1:[q1min,q1max], q2:[q2min,q2max]
    
    Parameters:
    -----------
    q1min : float
        Minimum value for first dimension (q1)
    q1max : float
        Maximum value for first dimension (q1)
    q2min : float
        Minimum value for second dimension (q2)
    q2max : float
        Maximum value for second dimension (q2)
    n : int
        Number of samples to generate
    
    Returns:
    --------
    q : numpy array of shape (n, 2)
        Latin Hypercube Samples where each row is a sample [q1, q2]
        
    Example:
    --------
    q = lhsSampling_py(1.5, 5.0, 12, 25, 20)
    # Returns 20 samples with q1 in [1.5, 5.0] and q2 in [12, 25]
    """
    print(f"Generating {n} LHS samples for q1:[{q1min},{q1max}], q2:[{q2min},{q2max}]")
    
    # Initialize array to store samples: n rows (samples), 2 columns (dimensions)
    q = np.zeros((n, 2))
    
    # Define ranges for the two dimensions
    ranges = [(q1min, q1max), (q2min, q2max)]
    n_dimensions = 2
    
    for dim in range(n_dimensions):
        """
        generate LHS samples for this dimension
        so the intervals are divided into n parts, for the normalized [0,1] 
        range we have n intervals (+1 because we start at 0 btw)
        
        so if n=5 and range is 0-1, intervals are:
        0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        """
        intervals = np.linspace(0, 1, n + 1)
        
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
        q[:, dim] = min_val + lhs_samples * (max_val - min_val) #all rows (samples), but only column (dimension)
    
    return q

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
q1min = 1.5   #q1 #p belongs [1.5,5]
q1max = 5.0
q2min = 12    #q2 #t belongs [12,25]
q2max = 25
n = 20

#usage - call the function with the new signature
q = lhsSampling_py(q1min, q1max, q2min, q2max, n)

# For plotting, we need to create ranges in the old format
ranges = [(q1min, q1max), (q2min, q2max)]
plot_and_display_samples(q, ranges)