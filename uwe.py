#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cvxpy as cp
import numpy as np

def gaussian_randomization(U, num_samples=100):
    """Extract a feasible rank-1 solution from the SDR result."""
    eigvals, eigvecs = np.linalg.eigh(U)
    idx = np.argmax(eigvals)  # Pick the largest eigenvalue
    u_approx = eigvecs[:, idx] * np.sqrt(eigvals[idx])
    return u_approx

# Parameters (example values)
K, M = 3, 3  # Number of users and RIS elements
w_c = cp.Variable((K, 1), complex=True)
w_r = cp.Variable((M, 1), complex=True)
v1 = cp.Variable((M, 1), complex=True)
v2 = cp.Variable((M, 1), complex=True)

# Initial values
w_c.value = np.random.randn(K, 1) + 1j * np.random.randn(K, 1)
w_r.value = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
v1.value = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
v2.value = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)

epsilon = 1e-3  # Convergence tolerance
max_iter = 100  # Max iterations
nu = 0

while nu < max_iter:
    # Solve SOCP problem (Example: Minimize transmit power)
    objective = cp.Minimize(cp.norm(w_c, 'fro')**2 + cp.norm(w_r, 'fro')**2)
    constraints = [cp.norm(w_c, 'fro')**2 + cp.norm(w_r, 'fro')**2 <= 10]  # Example constraint to ensure feasibility
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)  # Switched to CVXOPT solver for better compatibility
    
    # Update v1 using SDR
    U1 = cp.Variable((M, M), hermitian=True)
    constraints_sdr = [U1 >> 0, cp.trace(U1) <= 10]  # Adding an upper bound constraint for feasibility
    prob_sdr = cp.Problem(cp.Minimize(cp.trace(U1)), constraints_sdr)
    prob_sdr.solve(solver=cp.CVXOPT)
    
    # Extract u1 using Gaussian Randomization
    if U1.value is not None:  # Ensure a solution exists before updating v1
        v1.value = gaussian_randomization(U1.value)
    
    # Update v2 by solving another convex problem
    prob_v2 = cp.Problem(cp.Minimize(cp.norm(v2, 'fro')), [cp.norm(v2, 'fro') <= 5])  # Added feasibility constraint
    prob_v2.solve(solver=cp.CVXOPT)
    
    # Compute total transmit power
    P_b_new = np.linalg.norm(w_c.value)**2 + np.linalg.norm(w_r.value)**2
    
    # Convergence check
    if nu > 0 and abs(P_b_new - P_b_old) < epsilon:
        break
    
    P_b_old = P_b_new
    nu += 1

