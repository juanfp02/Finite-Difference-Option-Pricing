from FDM_base import FDMBase
from helper_Functions import solve_tridiagonal
import numpy as np



"""

FDM solvers (all 4 methods) Classes
JFPerez
July '24

"""


# Euler Explicit Method
class ExplicitFDM(FDMBase):
    def __init__(self, pde, N, Nj):
        super().__init__(pde, N, Nj)

    def solve(self):
        # coefficients
        sigma, r, q = self._pde._option.sigma, self._pde._option.r, self._pde._option.q
        nu = r - q - 0.5 * sigma ** 2
        pu = 0.5 * self._dt * ((sigma / self._dx) ** 2 + nu / self._dx)
        pm = 1.0 - self._dt * ((sigma / self._dx) ** 2 + r)
        pd = 0.5 * self._dt * ((sigma / self._dx) ** 2 - nu / self._dx)

        # Set terminal payoff
        for j in range(2 * self._Nj + 1):
            self._grid[self._N, j] = self._pde._option.payoff(self._asset_prices[j])

        # Backward induction
        for i in range(self._N - 1, -1, -1):
            for j in range(1, 2 * self._Nj):
                self._grid[i, j] = pu * self._grid[i + 1, j + 1] + pm * self._grid[i + 1, j] + pd * self._grid[i + 1, j - 1]
            
            # Boundary conditions
            self._grid[i, 0] = self._grid[i, 1]
            self._grid[i, 2 * self._Nj] = self._grid[i, 2 * self._Nj - 1] + (self._asset_prices[2 * self._Nj] - self._asset_prices[2 * self._Nj - 1])
        
        return self._grid[0, self._Nj]




#Euler Implicit Method

class ImplicitFDM(FDMBase):
    def __init__(self, pde, N, Nj):
        super().__init__(pde, N, Nj)

    def solve(self):
        # coefficients
        sigma, r, q = self._pde._option.sigma, self._pde._option.r, self._pde._option.q
        nu = r - q - 0.5 * sigma ** 2
        pu = -0.5 * self._dt * ((sigma / self._dx) ** 2 + nu / self._dx)
        pm = 1.0 + self._dt * ((sigma / self._dx) ** 2 + r)
        pd = -0.5 * self._dt * ((sigma / self._dx) ** 2 - nu / self._dx)

        # Set terminal payoff
        grid = np.zeros(2 * self._Nj + 1)
        for j in range(2 * self._Nj + 1):
            grid[j] = self._pde._option.payoff(self._asset_prices[j])

        # Boundary conditions
        if self._pde._option.option_type == 'Call':
            lambdaU = self._asset_prices[2 * self._Nj] - self._asset_prices[2 * self._Nj - 1]
            lambdaL = 0.0
        elif self._pde._option.option_type == 'Put':
            lambdaU = 0.0
            lambdaL = -1.0 * (self._asset_prices[1] - self._asset_prices[0])

        # Backwards induction
        for i in range(self._N):
            grid = solve_tridiagonal(grid, pu, pm, pd, lambdaL, lambdaU, self._Nj, 'Implicit')
        
        return grid[self._Nj]


#Crank Nicolson Method

class Crank_NicolsonFDM(FDMBase):
    def __init__(self, pde, N, Nj):
        super().__init__(pde, N, Nj)

    def solve(self):
        # coefficients
        sigma, r, q = self._pde._option.sigma, self._pde._option.r, self._pde._option.q
        nu = r - q - 0.5 * sigma ** 2
        pu = -0.25 * self._dt * ((sigma / self._dx) ** 2 + nu / self._dx)
        pm = 1.0 + 0.5*self._dt * ((sigma / self._dx) ** 2) + 0.5*r*self._dt
        pd = -0.25 * self._dt * ((sigma / self._dx) ** 2 - nu / self._dx)

        # Set terminal payoff
        grid = np.zeros(2 * self._Nj + 1)
        for j in range(2 * self._Nj + 1):
            grid[j] = self._pde._option.payoff(self._asset_prices[j])

        # Boundary conditions
        if self._pde._option.option_type == 'Call':
            lambdaU = self._asset_prices[2 * self._Nj] - self._asset_prices[2 * self._Nj - 1]
            lambdaL = 0.0
        elif self._pde._option.option_type == 'Put':
            lambdaU = 0.0
            lambdaL = -1.0 * (self._asset_prices[1] - self._asset_prices[0])

        # Backwards induction
        for i in range(self._N):
            grid = solve_tridiagonal(grid, pu, pm, pd, lambdaL, lambdaU, self._Nj, 'Crank Nicolson')
        
        return grid[self._Nj]
    
class RungeKuttaFDM(FDMBase):
    def __init__(self, pde, N, Nj):
        super().__init__(pde, N, Nj)

    def solve(self):
        # coefficients
        sigma, r, q = self._pde._option.sigma, self._pde._option.r, self._pde._option.q
        nu = r - q - 0.5 * sigma ** 2
        dx2 = self._dx ** 2
        dt = self._dt

        # Set terminal payoff
        grid = np.zeros(2 * self._Nj + 1)
        for j in range(2 * self._Nj + 1):
            grid[j] = self._pde._option.payoff(self._asset_prices[j])

        # Backwards induction using Runge-Kutta method
        for i in range(self._N):
            new_grid = np.zeros(2 * self._Nj + 1)
            for j in range(1, 2 * self._Nj):
                f1 = self._diffusion(grid, j) + self._convection(grid, j) + self._zero(grid, j)
                k1 = dt * f1
                f2 = self._diffusion(grid + 0.5 * k1, j) + self._convection(grid + 0.5 * k1, j) + self._zero(grid + 0.5 * k1, j)
                k2 = dt * f2
                f3 = self._diffusion(grid + 0.5 * k2, j) + self._convection(grid + 0.5 * k2, j) + self._zero(grid + 0.5 * k2, j)
                k3 = dt * f3
                f4 = self._diffusion(grid + k3, j) + self._convection(grid + k3, j) + self._zero(grid + k3, j)
                k4 = dt * f4
                new_grid[j] = grid[j] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            
            # Update boundary conditions
            new_grid[0] = new_grid[1]
            new_grid[2 * self._Nj] = new_grid[2 * self._Nj - 1] + (self._asset_prices[2 * self._Nj] - self._asset_prices[2 * self._Nj - 1])
            grid = new_grid
        
        return grid[self._Nj]

    def _diffusion(self, grid, j):
        sigma = self._pde._option.sigma
        return 0.5 * (sigma ** 2) * ((grid[j + 1] - 2 * grid[j] + grid[j - 1]) / self._dx ** 2)

    def _convection(self, grid, j):
        r, q = self._pde._option.r, self._pde._option.q
        nu = r - q - 0.5 * self._pde._option.sigma ** 2
        return nu * (grid[j + 1] - grid[j - 1]) / (2 * self._dx)

    def _zero(self, grid, j):
        r = self._pde._option.r
        return -r * grid[j]











"""

FDM solvers (all 4 methods) Classes
JFPerez
July '24

"""