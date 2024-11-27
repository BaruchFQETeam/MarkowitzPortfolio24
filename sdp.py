import pandas as pd
import numpy as np
import cvxpy as cp

# Load your data
df = pd.read_csv("sp500_5year_close_prices.csv")
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
rho = 0.01  # Adjust based on desired sparsity
nu = 0.001  # Minimum variance threshold

# Objective function
objective = cp.Minimize(cp.trace(M @ Y) + rho * cp.sum(cp.abs(Y)))

# Constraints
constraints = [
    cp.trace(A_0 @ Y) >= nu,   # Variance constraint
    cp.trace(Y) == 1,           # Normalization constraint
    Y >= 0 
]

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


eigenvalues, eigenvectors = np.linalg.eigh(Y.value)

max_index = np.argmax(eigenvalues)


highest_eigenvalue = eigenvalues[max_index]


corresponding_eigenvector = eigenvectors[:, max_index]

print("Highest Eigenvalue:")
print(highest_eigenvalue)

print("Corresponding Eigenvector:")
print(corresponding_eigenvector)


