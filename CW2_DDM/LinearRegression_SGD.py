# Linear Regression using Stochastic Gradient Descent (SGD)
# AERO40041 Coursework 2 - Task 1
#
# This submission was prepared by Juan Ignacio Doval Roque [10752534] and Sacha Muller [10873681].
#
# NOTE: Plots have been hard coded for given data (Qdot vs dT), therefore, if other data is used,
# the plotting section may need to be adjusted accordingly. Also the path to the data file should be changed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

your_path = 'CW2_DDM/'

def load_and_prepare_data(filename):
    """
    Load dataset from CSV and prepare feature matrix X and target vector y.
    Adds a column of ones to X, for the bias term.
    """
    data = pd.read_csv(filename)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((ones, X), axis=1)
    
    return X, y

def compute_mse_loss(X, y, w):
    """
    Compute Mean Squared Error loss.
    
    MSE = (1/N) * sum((y_pred - y)^2)
    """
    N = X.shape[0]
    y_pred = X @ w
    error = y_pred - y
    loss = (1/N) * np.sum(error ** 2)
    return loss

def sgd_linear_regression(X, y, learning_rate=0.01, num_epochs=100, seed=42):
    """
    Train linear regression using Stochastic Gradient Descent.
    
    In SGD, we update weights after each individual sample.
    One EPOCH means we have processed every sample exactly once.
    
    Parameters:
    -----------
    X : numpy array of shape (N, D+1)
        Feature matrix with bias column
    y : numpy array of shape (N, 1)
        Target values
    learning_rate : float
        Step size for gradient updates
    num_epochs : int
        Number of complete passes through the dataset
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    w : numpy array
        Learned weights
    loss_history : list
        Loss value recorded at the end of each epoch
    """
    np.random.seed(seed)
    
    N, D = X.shape
    w = np.random.randn(D, 1) * 0.01
    loss_history = []
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(N)
        
        for i in indices:
            xi = X[i:i+1, :]
            yi = y[i:i+1, :]
            y_pred = xi @ w
            error = y_pred - yi
            gradient = 2 * xi.T @ error
            w = w - learning_rate * gradient
        
        current_loss = compute_mse_loss(X, y, w)
        loss_history.append(current_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {current_loss:.6f}")
    
    return w, loss_history

def predict(X, w):
    """Make predictions using learned weights."""
    return X @ w

if __name__ == "__main__":
    
    X, y = load_and_prepare_data(your_path + "window_heat.csv") 
    
    dT_original = X[:, 1].copy()
    
    print("DATA CHECK:")
    print(f"dT range: {dT_original.min():.2f} to {dT_original.max():.2f} °C")
    print(f"Qdot range: {y.min():.2f} to {y.max():.2f} W")
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]-1} features")
    
    X_normalized = X.copy()
    means = X[:, 1:].mean(axis=0)
    stds = X[:, 1:].std(axis=0)
    stds[stds == 0] = 1 
    X_normalized[:, 1:] = (X[:, 1:] - means) / stds
    
    print("\nStart of SGD Linear Regression:")
    
    w_sgd, loss_history = sgd_linear_regression(
        X_normalized, y,
        learning_rate=0.01,
        num_epochs=100,
        seed=42
    )

    print(f"\nFinal Loss (MSE): {loss_history[-1]:.6f}")
    print(f"Learned weights (normalized): {w_sgd.flatten()}")
    
    y_pred = predict(X_normalized, w_sgd)
    
    final_mse = np.mean((y - y_pred) ** 2)
    final_rmse = np.sqrt(final_mse)
    print(f"Final RMSE: {final_rmse:.6f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Plot 1: Loss vs Epoch
    axes[0].plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('SGD Training: Loss vs Epoch', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Qdot vs dT with fitted line
    axes[1].scatter(dT_original, y, alpha=0.6, edgecolors='black', 
                    linewidth=0.5, label='Data points', color='blue')
    dT_line = np.linspace(dT_original.min(), dT_original.max(), 100)
    dT_line_normalized = (dT_line - means[0]) / stds[0]
    X_line = np.column_stack([np.ones(100), dT_line_normalized])
    y_line = X_line @ w_sgd
    
    axes[1].plot(dT_line, y_line, 'r-', linewidth=2, label='Fitted line')
    axes[1].set_xlabel('ΔT (°C)', fontsize=12)
    axes[1].set_ylabel('$\dot{Q}$ (W)', fontsize=12)
    axes[1].set_title('Heat Loss vs Temperature Difference', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs Actual
    axes[2].scatter(y, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', 
                 linewidth=2, label='Perfect Prediction')
    axes[2].set_xlabel('Actual $\dot{Q}$ (W)', fontsize=12)
    axes[2].set_ylabel('Predicted $\dot{Q}$ (W)', fontsize=12)
    axes[2].set_title('Predictions vs Actual Values', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(your_path + 'LinearRegression_SGD_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as '{your_path}LinearRegression_SGD_output.png'")