import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


"""

Helper functions
JFPerez
July '24

"""


def solve_tridiagonal(coefficients, upper, main, lower, lower_bound, upper_bound, grid_points, method):
    solution = np.zeros(2 * grid_points + 1)
    adjusted_main = [main + lower]

    # Thomas algorithm for tridiagonal system
    if method == 'Implicit':
        adjusted_rhs = [coefficients[1] + lower * lower_bound]

        # Forward sweep
        for j in range(2, 2 * grid_points):
            adjusted_main.append(main - upper * lower / adjusted_main[j - 2])
            adjusted_rhs.append(coefficients[j] - adjusted_rhs[j - 2] * lower / adjusted_main[j - 2])

        # Backward substitution
        solution[2 * grid_points] = (adjusted_rhs[-1] + adjusted_main[-1] * upper_bound) / (upper + adjusted_main[-1])
        solution[2 * grid_points - 1] = solution[2 * grid_points] - upper_bound

        for j in range(2 * grid_points - 2, -1, -1):
            solution[j] = (adjusted_rhs[j - 1] - upper * solution[j + 1]) / adjusted_main[j - 1]

        solution[0] = solution[1] - lower_bound
        return solution

    elif method == 'Crank Nicolson':
        adjusted_rhs = [-upper * coefficients[2] - (main - 2) * coefficients[1] - lower * coefficients[0] + lower * lower_bound]

        # Forward sweep
        for j in range(2, 2 * grid_points):
            adjusted_main.append(main - upper * lower / adjusted_main[j - 2])
            adjusted_rhs.append(-upper * coefficients[j + 1] - (main - 2) * coefficients[j] - lower * coefficients[j - 1] - adjusted_rhs[j - 2] * lower / adjusted_main[j - 2])

        # Backward substitution
        solution[2 * grid_points] = (adjusted_rhs[-1] + adjusted_main[-1] * upper_bound) / (upper + adjusted_main[-1])
        solution[2 * grid_points - 1] = solution[2 * grid_points] - upper_bound

        for j in range(2 * grid_points - 2, 0, -1):
            solution[j] = (adjusted_rhs[j - 1] - upper * solution[j + 1]) / adjusted_main[j - 1]

        solution[0] = coefficients[0]
        return solution


def plot_convergence_analysis(N_values, prices, analytical_price):
    fig = plt.figure(figsize=(16, 8))
    plt.plot(N_values, prices, label='FDM Price')
    plt.axhline(y=analytical_price, color='r', linestyle='-', label='Analytical Price')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Number of Time Steps (N)')
    plt.ylabel('Option Price')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)





"""

Helper Functions
JFPerez
July '24

"""