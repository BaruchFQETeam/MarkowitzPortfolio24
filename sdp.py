import pandas as pd
import numpy as np
import cvxpy as cp

# Load your data
df = pd.read_csv("sp500_5year_close_prices.csv")
#df = df[['Date', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL']]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Use returns for stationarity
returns = df.pct_change().dropna()
x_t = returns.values

# Center the data
x_bar = np.mean(x_t, axis=0)
x_tilde = x_t - x_bar

# Compute autocovariance matrices
T = x_tilde.shape[0]
A_0 = (x_tilde.T @ x_tilde) / T

x_tilde_t = x_tilde[:-1]
x_tilde_t_plus_1 = x_tilde[1:]
A_1 = (x_tilde_t.T @ x_tilde_t_plus_1) / (T - 1)

# Compute M
A0_inv = np.linalg.pinv(A_0)  
M = A_1 @ A0_inv @ A_1.T


'------------------------------------------------------------------------------------------------'

n = A_0.shape[0]  # Number of assets

# Define the variable Y (n x n symmetric matrix)
Y = cp.Variable((n, n), PSD=True)

# Define the regularization parameter
rho = 0.1  # Adjust based on desired sparsity
nu = 0.1  # Minimum variance threshold

# Objective function
objective = cp.Minimize(cp.trace(M @ Y) + rho * cp.sum(cp.abs(Y)))

# Constraints
constraints = [
    cp.trace(A_0 @ Y) >= nu,   # Variance constraint
    cp.trace(Y) == 1,           # Normalization constraint
    Y >= 0 #is implied by PSD=True
]

#max_weight = 0.1  # Maximum allowable weight per asset
#for i in range(n):
#    constraints.append(Y[i, i] <= max_weight**2)
    
# Define the problem
prob = cp.Problem(objective, constraints)

'------------------------------------------------------------------------------------------------'
# Solve the problem
result = prob.solve(solver=cp.SCS)

# Check if the problem was solved successfully
if prob.status not in ["infeasible", "unbounded"]:
    print("Problem solved successfully!")
else:
    print(f"Problem status: {prob.status}")

import scipy.linalg

# Ensure Y is symmetric
Y_value = Y.value
Y_symmetric = (Y_value + Y_value.T) / 2

# Eigenvalue decomposition
eigvals, eigvecs = scipy.linalg.eigh(Y_symmetric)

# Find the largest eigenvalue
idx = np.argmax(eigvals)

y_optimal = eigvecs[:, idx]
y_optimal = y_optimal * np.sign(np.sum(y_optimal))


# Normalize y_optimal
y_optimal = y_optimal / np.sum(y_optimal)

# Display the weights
for asset, weight in zip(returns.columns, y_optimal):
    print(f"{asset}: {weight:.4f}")

# Check sparsity
non_zero_weights = np.sum(np.abs(y_optimal) > 1e-5)
print(f"Number of non-zero weights: {non_zero_weights}")














