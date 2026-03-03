from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
from scipy.stats import qmc

theta_params = [1.0, 2.0, 0.5]

# Task 1.2(b): Implement the model function
def model_function(q, theta):
    """
    Evaluate the model function f(q; θ) at given q values.
    
    Parameters:
    -----------
    q : array-like
        Input values (can be scalar or array)
    theta : list or array
        Parameters [theta1, theta2, theta3]
    
    Returns:
    --------
    array-like
        Function values f(q; θ)
    """
    theta1, theta2, theta3 = theta
    q = np.asarray(q)
    
    print(f"Evaluating model function with theta: {theta}")
    print(f"Input q values: {q}")
    
    term1 = theta1 * q**2
    term2 = theta2 * np.sin(q)**2
    term3 = theta3 * (q**3 /(q**2 + 1))
    
    return term1 + term2 + term3

#EXAMPLE USAGE
# q_values = np.linspace(0, 1, 10)

# f_values = model_function(q_values, theta_params)

# plt.plot(q_values, f_values, label='f(q; θ)')
# plt.xlabel('q')
# plt.ylabel('f(q; θ)')
# plt.title('Model Function Evaluation')
# plt.legend()
# plt.show()

# Task 1.2(c): Generate random samples and evaluate
def evaluate_random_samples(theta, n_uniform=100, n_normal=100):
    """
    Generate random samples from uniform and normal distributions
    and evaluate the model function.
    
    Parameters:
    -----------
    theta : list or array
        Parameters [theta1, theta2, theta3]
    n_uniform : int
        Number of uniform samples
    n_normal : int
        Number of normal samples
        Number of normal samples
    
    Returns:
    --------
    dict
        Dictionary containing samples and function values
    """
    # Generate random samples
    q_uniform = np.random.uniform(-3, 5, n_uniform)
    q_normal = np.random.normal(1, 2, n_normal)  # mean=1, std=sqrt(4)=2
    
    # Evaluate function
    f_uniform = model_function(q_uniform, theta)
    f_normal = model_function(q_normal, theta)
    
    return {
        'q_uniform': q_uniform,
        'f_uniform': f_uniform,
        'q_normal': q_normal,
        'f_normal': f_normal
    }

def plot_function_comparison(theta, random_results, n_test=200):
    """
    Plot the function values at random samples and compare with
    equi-spaced test samples.
    
    Parameters:
    -----------
    theta : list or array
        Parameters [theta1, theta2, theta3]
    random_results : dict
        Results from evaluate_random_samples
    n_test : int
        Number of equi-spaced test samples
    """
    # Generate equi-spaced test samples
    q_test = np.linspace(-5, 7, n_test)
    f_test = model_function(q_test, theta)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot test samples (equi-spaced) as a line
    plt.plot(q_test, f_test, 'k-', linewidth=2, label='Equi-spaced test samples $q^*$', zorder=1)
    
    # Plot random samples as scatter points
    plt.scatter(random_results['q_uniform'], random_results['f_uniform'], 
                c='blue', alpha=0.6, s=30, label='Random samples: $q \sim \mathcal{U}[-3, 5]$', zorder=2)
    plt.scatter(random_results['q_normal'], random_results['f_normal'], 
                c='red', alpha=0.6, s=30, label='Random samples: $q \sim \mathcal{N}(1, 4)$', zorder=2)
    
    plt.xlabel('$q$', fontsize=12)
    plt.ylabel('$f(q; \\theta)$', fontsize=12)
    plt.title('Model Function Evaluation: Random vs Equi-spaced Samples', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return q_test, f_test

# results = evaluate_random_samples(theta_params, n_uniform=100, n_normal=100)
# plot_function_comparison(theta_params, results)

def evaluate_noisy_function(theta, n_uniform=100, n_normal=100, sigma_noise=0.5):
    """
    Evaluate a noisy version of the model function with Gaussian noise.
    
    Parameters:
    -----------
    theta : list or array
        Parameters [theta1, theta2, theta3]
    n_uniform : int
        Number of uniform samples
    n_normal : int
        Number of normal samples
    sigma_noise : float
        Standard deviation of Gaussian noise
    
    Returns:
    --------
    dict
        Dictionary containing samples and noisy function values
    """
    # Generate random samples
    q_uniform = np.random.uniform(-3, 5, n_uniform)
    q_normal = np.random.normal(1, 2, n_normal)
    
    # Evaluate function
    f_uniform = model_function(q_uniform, theta)
    f_normal = model_function(q_normal, theta)
    
    # Add noise only to random training samples
    noise_uniform = np.random.normal(0, sigma_noise, n_uniform)
    noise_normal = np.random.normal(0, sigma_noise, n_normal)
    
    y_uniform = f_uniform + noise_uniform
    y_normal = f_normal + noise_normal
    
    return {
        'q_uniform': q_uniform,
        'y_uniform': y_uniform,
        'f_uniform': f_uniform,  # Clean values for reference
        'q_normal': q_normal,
        'y_normal': y_normal,
        'f_normal': f_normal  # Clean values for reference
    }
    
def plot_noisy_function_comparison(theta, noisy_results, n_test=200):
    """
    Plot the noisy function values at random samples and compare with
    clean equi-spaced test samples.
    
    Parameters:
    -----------
    theta : list or array
        Parameters [theta1, theta2, theta3]
    noisy_results : dict
        Results from evaluate_noisy_function
    n_test : int
        Number of equi-spaced test samples
    """
    # Generate equi-spaced test samples (no noise added)
    q_test = np.linspace(-3, 5, n_test)
    f_test = model_function(q_test, theta)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot test samples (equi-spaced, no noise) as a line
    plt.plot(q_test, f_test, 'k-', linewidth=2, label='Clean test samples $q^*$', zorder=1)
    
    # Plot noisy random samples as scatter points
    plt.scatter(noisy_results['q_uniform'], noisy_results['y_uniform'], 
                c='blue', alpha=0.6, s=30, label='Noisy samples: $q \sim \mathcal{U}[-3, 5]$', zorder=2)
    plt.scatter(noisy_results['q_normal'], noisy_results['y_normal'], 
                c='red', alpha=0.6, s=30, label='Noisy samples: $q \sim \mathcal{N}(1, 4)$', zorder=2)
    
    plt.xlabel('$q$', fontsize=12)
    plt.ylabel('$y = f(q; \\theta) + \\varepsilon$', fontsize=12)
    plt.title('Noisy Function Evaluation: Random Samples with Noise vs Clean Test Samples', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return q_test, f_test

# noisy_results = evaluate_noisy_function(theta_params, n_uniform=100, n_normal=100, sigma_noise=0.5)
# plot_noisy_function_comparison(theta_params, noisy_results)

def latin_hypercube_sampling(n, bounds=None):
    """
    Generate n joint samples using Latin Hypercube Sampling (LHS)
    for a 2D design space.
    
    Parameters:
    -----------
    n : int
        Number of samples to generate
    bounds : list of tuples, optional
        Bounds for each dimension [(q1_min, q1_max), (q2_min, q2_max)]
        Default: [(-2, 5), (1, 3)]
    
    Returns:
    --------
    array
        Array of shape (n, 2) containing LHS samples
    """
    if bounds is None:
        bounds = [(-2, 5), (1, 3)]  # Default bounds for q1 and q2
    
    # Create LHS sampler for 2D space
    sampler = qmc.LatinHypercube(d=2, seed=None)
    
    # Generate samples in [0, 1]^2
    samples_unit = sampler.random(n=n)
    
    # Scale samples to desired bounds
    q1_min, q1_max = bounds[0]
    q2_min, q2_max = bounds[1]
    
    samples = np.zeros_like(samples_unit)
    samples[:, 0] = q1_min + (q1_max - q1_min) * samples_unit[:, 0]
    samples[:, 1] = q2_min + (q2_max - q2_min) * samples_unit[:, 1]
    
    return samples

def visualize_lhs_samples(samples, bounds=None):
    """
    Visualize the LHS samples in 2D space.
    
    Parameters:
    -----------
    samples : array
        LHS samples of shape (n, 2)
    bounds : list of tuples, optional
        Bounds for each dimension
    """
    if bounds is None:
        bounds = [(-2, 5), (1, 3)]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', s=50, alpha=0.6, edgecolors='black')
    plt.xlabel('$q_1$', fontsize=12)
    plt.ylabel('$q_2$', fontsize=12)
    plt.title(f'Latin Hypercube Sampling: {len(samples)} samples', fontsize=14)
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

samples = latin_hypercube_sampling(100, bounds=None)
visualize_lhs_samples(samples, bounds=None)