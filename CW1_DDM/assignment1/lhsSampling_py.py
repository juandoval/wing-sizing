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
    
    q = np.zeros((n, 2))
    
    ranges = [(q1min, q1max), (q2min, q2max)]
    n_dimensions = 2
    
    for dim in range(n_dimensions):

        intervals = np.linspace(0, 1, n + 1)
        lhs_samples = np.random.uniform(intervals[:-1], intervals[1:])
        
        np.random.shuffle(lhs_samples)
        min_val, max_val = ranges[dim] 
        q[:, dim] = min_val + lhs_samples * (max_val - min_val)
    
    return q

def plot_and_display_samples(samples, ranges, title="Latin Hypercube Sampling (LHS) for Airfoil Parameters"):
    """
    Plot samples and display them in a table format.
    
    Parameters:
    samples: numpy array of samples
    ranges: list of tuples for axis limits
    title: plot title
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(samples[:, 0], samples[:, 1], color='purple', s=80, alpha=0.7, edgecolors='#6b2c91') 
    plt.xlim(ranges[0])
    plt.ylim(ranges[1])
    plt.xlabel('p (Location of Max Camber)')
    plt.ylabel('t (Airfoil Half Thickness)') 
    plt.title(title)
    plt.grid(True, alpha=0.3) 
    
    for i, (x, y) in enumerate(samples):
        plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    table_data = []
    for i, sample in enumerate(samples):
        table_data.append([f"Sample {i+1}", f"{sample[0]:.4f}", f"{sample[1]:.4f}"])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Sample #', 'X Value', 'Y Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0: 
                cell.set_facecolor('#e2b525')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    plt.show()
    
    df = pd.DataFrame(samples, columns=['X Parameter', 'Y Parameter'])
    df.index = [f'Sample {i+1}' for i in range(len(samples))]
    print("\nLatin Hypercube Samples:")
    print(df.round(4))


# Example usage and plotting
q1min = 1.5  
q1max = 5.0
q2min = 12   
q2max = 25
n = 20

q = lhsSampling_py(q1min, q1max, q2min, q2max, n)

ranges = [(q1min, q1max), (q2min, q2max)]
plot_and_display_samples(q, ranges)