import numpy as np
from helper_Functions import normalize



"""

FDM Base Class
JFPerez
July '24

"""

class FDMBase:
    def __init__(self, pde, N, Nj):
        self._pde = pde
        self._N = N
        self._Nj = Nj

        self._dt = self._pde._option.T / N
        self._dx = self._pde._option.sigma * np.sqrt(3 * self._dt)
        self._grid = np.zeros((N + 1, 2 * Nj + 1))
        self._asset_prices = self._initialize_asset_prices()

    def _initialize_asset_prices(self):
        S = self._pde._option.S
        St = [S * np.exp(-self._Nj * self._dx)]
        for j in range(1, 2 * self._Nj + 1):
            St.append(St[j - 1] * np.exp(self._dx))
        return St
        
    def Surface_data(self):
        T = self._pde._option.T
        S_values = np.array(self._asset_prices)  # Asset prices
        time_steps = np.arange(self._N + 1)  # Time steps from 0 to N

        # Option values from the grid
        option_values = self._grid

        # Define boundaries for asset price and option value
        max_asset_price = 2 * self._pde._option.S
        max_option_value = 2 * np.max(option_values)

        # Clip values to the defined maximum boundaries
        clipped_option_values = np.clip(option_values, 0, max_option_value)
        clipped_asset_values = np.clip(S_values, 0, max_asset_price)

        # Convert time steps to T - t (remaining time)
        time_remaining = T - (time_steps * self._dt)
        normalized_t_values = time_remaining / T 

        # Meshgrid for 3D plotting
        S_mesh, T_mesh = np.meshgrid(clipped_asset_values, normalized_t_values)
        
        # Interpolating option values for each time step
        C_mesh = np.zeros_like(S_mesh)
        for i in range(len(normalized_t_values)):
            C_mesh[i, :] = np.interp(S_mesh[i, :], S_values, clipped_option_values[i, :])

        return S_mesh, T_mesh, C_mesh











"""

FDM Base Class
JFPerez
July '24

"""