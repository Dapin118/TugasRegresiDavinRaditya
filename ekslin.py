import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import os

# Define the file path
file_path = r"C:\Users\davin\OneDrive\Dokumen\metnum\Student_Performance.csv"

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
else:
    # Load data
    data = pd.read_csv(file_path)

    # Select relevant columns
    data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

    # Extract data for analysis
    X = data['Hours Studied'].values
    y = data['Performance Index'].values

    # Exponential Model (Method 3)
    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    # Fit exponential model to data
    params, covariance = curve_fit(exponential_model, X, y)
    a, b = params
    y_pred_exponential = exponential_model(X, a, b)

    # Linear Model
    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    y_pred_linear = slope * X + intercept

    # Plot data and regression results
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred_exponential, color='green', label='Exponential Regression')
    plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    plt.title('Regression Models')
    plt.legend()
    plt.show()

    # Calculate RMS error for exponential model
    rms_exponential = np.sqrt(mean_squared_error(y, y_pred_exponential))
    print(f'RMS Error for Exponential Model: {rms_exponential}')

    # Calculate RMS error for linear model
    rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
    print(f'RMS Error for Linear Model: {rms_linear}')
