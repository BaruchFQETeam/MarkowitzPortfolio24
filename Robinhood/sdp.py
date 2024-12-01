import pandas as pd
import numpy as np
import cvxpy as cp

def get_optimal_weights():
    """ Returns the optimal weights for the given portfolio in a dictionary format. """
    # Load your data
    df = pd.read_csv("StockPortfolio_5year_close_prices.csv")

    #df = df[['Date','AMGN', 'CRWD', 'CARR', 'BA', 'TSLA', 'VRTX', 'GE', 'PLTR', 'ABNB']]

    df = df.drop(["SPY"], axis=1)

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
    A0_inv = np.linalg.pinv(A_0 + 1e-6 * np.eye(A_0.shape[0]))
    M = A_1 @ A0_inv @ A_1.T


    '------------------------------------------------------------------------------------------------'

    n = A_0.shape[0]  # Number of assets

    # Define the variable Y (n x n symmetric matrix)
    Y = cp.Variable((n, n), PSD=True)

    # Define the regularization parameter
    rho = 0.1  # Adjust based on desired sparsity
    nu = 0.001  # Minimum variance threshold

    # Objective function
    #objective = cp.Minimize(cp.trace(M @ Y) + rho * cp.sum(cp.abs(Y)))
    objective = cp.Minimize(cp.trace(M @ Y) + rho * cp.norm(Y, 'nuc'))

    # Constraints
    constraints = [
        cp.trace(A_0 @ Y) >= nu,   # Variance constraint
        cp.trace(Y) == 1,           # Normalization constraint
        Y >> 0 
    ]

    #min_weight = 0.01
    #constraints.append(Y >= np.diag(np.ones(n) * min_weight))

    # Define the problem
    prob = cp.Problem(objective, constraints)
    #prob = cp.Problem(objective)

    '------------------------------------------------------------------------------------------------'
    # Solve the problem
    result = prob.solve(solver=cp.SCS)

    #Check if the problem was solved successfullyif prob.status not in ["infeasible", "unbounded"]:
    if prob.status not in ["infeasible", "unbounded"]:
        print("Problem solved successfully!")
    else:
        print(f"Problem status: {prob.status}")


    eigenvalues, eigenvectors = np.linalg.eigh(Y.value)

    max_index = np.argmax(eigenvalues)

    corresponding_eigenvector = eigenvectors[:, max_index]
    highest_eigenvalue = eigenvalues[max_index]


    corresponding_eigenvector = np.abs(corresponding_eigenvector)

    print("Highest Eigenvalue:")
    print(highest_eigenvalue)

    print("Corresponding Eigenvector:")
    print(corresponding_eigenvector)
    y_optimal = corresponding_eigenvector / np.sum(corresponding_eigenvector)

    for asset, weight in zip(returns.columns, y_optimal*100):
            print(f"{asset}: {weight:.4f}")

    optimal_weights = {col: float(weight) for col, weight in zip(returns.columns, y_optimal)}
    cols = list(returns.columns)


    print(optimal_weights)
    print(cols)
    "dict(sorted(optimal_weights.items(), key=lambda item: item[1])))"
    return optimal_weights
