import cvxpy as cp
import numpy as np

# Define the variables
n = 3  # Dimension of the matrix
P = cp.Variable((n, n), symmetric=True)
Q = np.random.randn(n, n)  # Assume Q is a random symmetric matrix

# Define the Lyapunov inequality constraints
A = np.random.randn(n, n)  # Assume A is a random matrix
constraints = [A.T @ P + P @ A + Q << 0]  # A^T P + P A + Q < 0

# Define the objective function with a regularization term
regularization_term = cp.norm(P, 'fro')  # Frobenius norm of P
objective = cp.Minimize(regularization_term) 

# Create the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
problem.solve()

# Get the optimal solution
optimal_P = P.value

print("Optimal P:")
print(optimal_P)

# Verify the solution
eigenvalues, _ = np.linalg.eig(A.T @ optimal_P + optimal_P @ A + Q)
if np.all(eigenvalues < 0):
    print("Optimal P satisfies the Lyapunov inequality A^T P + P A + Q < 0.")
else:
    print("Optimal P does not satisfy the Lyapunov inequality A^T P + P A + Q < 0.")
