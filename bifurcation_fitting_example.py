import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def harvest(t, y, r, K, h, m):
    x, c = y
    dxdt = r * x * (1 - x / K) - c * (x**2 / (x**2 + h**2))
    dcdt = m
    return [dxdt, dcdt]

def evaluate_bifurcation_fitting():
    # 1. Load Data
    try:
        mat_data = sio.loadmat('Core Algorithms and Examples/Examples Bif Fitting/SleepDistance_Example.mat')
        tvec_plot = mat_data['tvec_plot'].flatten()
        xx_smooth = mat_data['xx_smooth'].flatten()
    except FileNotFoundError:
        print("Data files not found.")
        return

    # Initial parameters and condition
    # r, K, h, m
    K_init = np.mean(xx_smooth[:100])
    params = [3.0, K_init, 0.6, 0.45]
    x_ini = [np.mean(xx_smooth[:100]), 1.0]

    # Objective function for optimization
    def objective(p):
        r, K, h, m = p
        if r <= 0 or m <= 0 or K <= 0 or h <= 0:
            return np.inf
        # Solve ODE
        sol = solve_ivp(harvest, [tvec_plot[0], tvec_plot[-1]], x_ini, t_eval=tvec_plot, args=(r, K, h, m), method='RK45')
        if not sol.success or len(sol.y[0]) != len(tvec_plot):
            return np.inf
        # Compute MSE
        mse = np.mean((sol.y[0] - xx_smooth)**2)
        return mse

    print("Initial parameters:", params)
    print("Initial MSE:", objective(params))
    
    print("Optimizing parameters (this might take a few moments)...")
    bounds = [(0.5, 20.0), (K_init*0.5, K_init*1.5), (0.1, 2.0), (0.01, 2.0)]
    # Use Nelder-Mead for robustness on this type of problem
    res = minimize(objective, params, method='Nelder-Mead', bounds=bounds)
    
    if res.success:
        params_optim = res.x
        print("Optimization successful!")
    else:
        params_optim = params
        print("Optimization failed, using initial params.")
        
    print("Optimized parameters:", params_optim)
    print("Final MSE:", objective(params_optim))

    # Solve with optimized parameters
    r, K, h, m = params_optim
    sol = solve_ivp(harvest, [tvec_plot[0], tvec_plot[-1]], x_ini, t_eval=tvec_plot, args=(r, K, h, m))
    dd_x = sol.y[0]
    dd_c = sol.y[1]

    # 2. Bifurcation Tipping Point Evaluation (Roots of Cubic)
    # The equation is: -(r/K) * x^3 + r * x^2 - (c + r*h^2/K) * x + r*h^2 = 0
    c_vals = dd_c
    c_3sol = []
    roots_all = []

    for c in c_vals:
        # Coefficients of cubic ax^3 + bx^2 + cx + d = 0
        coeff_a = -(r / K)
        coeff_b = r
        coeff_c_term = -(c + r * h**2 / K)
        coeff_d = r * h**2
        
        roots = np.roots([coeff_a, coeff_b, coeff_c_term, coeff_d])
        real_roots = np.real(roots[np.isreal(roots) & (np.abs(np.imag(roots)) < 1e-5)])
        real_roots = real_roots[real_roots > 0]
        real_roots = np.sort(real_roots)
        
        if len(real_roots) == 3:
            c_3sol.append(c)
        
        roots_all.append(real_roots)

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot 1: Time Series Fit
    ax1.plot(tvec_plot, xx_smooth, color='k', linewidth=2, label='Data (Sleep Distance)')
    ax1.plot(tvec_plot, dd_x, color='b', linewidth=4, label='Bifurcation Fit')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('System state')
    ax1.set_xlim([-30, 10])
    ax1.axvline(0, color='r', linestyle='--', linewidth=2)
    
    if c_3sol:
        critical_c = c_3sol[-1]
        critical_idx = np.argmin(np.abs(c_vals - critical_c))
        critical_SValue = roots_all[critical_idx][1] # The middle root is the unstable branch separating basins
        ax1.axhline(critical_SValue, color='g', linestyle='--', linewidth=1.5, label='Critical S-Value')
    
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Bifurcation Diagram (c vs x)
    for i, c in enumerate(c_vals):
        rr = roots_all[i]
        if len(rr) == 1:
            ax2.plot(c, rr[0], 'k.', markersize=2)
        elif len(rr) == 3:
            ax2.plot(c, rr[0], 'k.', markersize=2)
            ax2.plot(c, rr[1], 'k.', markersize=2, alpha=0.5) # unstable branch
            ax2.plot(c, rr[2], 'k.', markersize=2)
            
    ax2.set_xlabel('Control variable (c)')
    ax2.set_ylabel('System state (x)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Core Algorithms and Examples/Examples Bif Fitting/Bifurcation_Python_Plot.png')
    plt.close(fig)

if __name__ == "__main__":
    evaluate_bifurcation_fitting()
