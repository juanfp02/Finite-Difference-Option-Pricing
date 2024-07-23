import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm


class Option:
    def __init__(self, S, K, T, r, q, sigma, option_type='Call'):
        self._S = S
        self._K = K
        self._T = T
        self._r = r
        self._q = q
        self._sigma = sigma
        self._option_type = option_type

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, value):
        self._option_type = value

    @property
    def retrieve_attributes(self):
        return self.S, self.K, self.T, self.r, self.sigma, self.option_type

    def payoff(self, x):
        if self.option_type == 'Call':
            return max(x - self.K, 0)
        elif self.option_type == 'Put':
            return max(self.K - x, 0)
        
    def black_scholes(self, S=None, t=None):

        if S is None:
            S = self.S            


        if t is None:
            T_t = self._T
        else:
            T_t = self._T - t

        d1 = (np.log(S / self._K) + (self._r - self._q + 0.5 * self._sigma ** 2) * T_t) / (self._sigma * np.sqrt(T_t))
        d2 = d1 - self._sigma * np.sqrt(T_t)

        if self._option_type == 'Call':
            price = S * np.exp(-self._q * T_t) * norm.cdf(d1) - self._K * np.exp(-self._r * T_t) * norm.cdf(d2)
        elif self._option_type == 'Put':
            price = self._K * np.exp(-self._r * T_t) * norm.cdf(-d2) - S * np.exp(-self._q * T_t) * norm.cdf(-d1)

        return price

    def boundary_condition(self, S, time):
        if self.option_type == 'Call':
            return max(0, S - self.K * np.exp(-self.r * (self.T - time)))
        elif self.option_type == 'Put':
            return max(0, self.K * np.exp(-self.r * (self.T - time)) - S)


class BlackScholesPDE:
    def __init__(self, option):
        self._option = option
        self.vol = option.sigma
        self.r = option.r

    def diffusion_coefficient(self, x):
        return 0.5 * (self.vol**2) * (x**2)
    
    def convection_coefficient(self, x):
        return self.r * x
    
    def zero_coefficient(self):
        return -self.r
    
    def source_coefficient(self):
        return 0.0
    
    def boundary_condition(self, S, time):
        return self._option.boundary_condition(S, time)



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

    
def plot_surface(S_mesh, T_mesh, C_mesh):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, T_mesh, C_mesh, cmap='viridis')

    ax.set_xlabel('Asset Price (S)')
    ax.set_ylabel('Time Remaining (T-t)')
    ax.set_zlabel('Option Price (C)')
    plt.title('Option Price Surface')
    st.pyplot(plt)



def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


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

def main():
    st.title("Option Pricing with Finite Difference Methods 📊")

    st.write("FDM are numerical techniques that can be used to approximate option values by discretizing the Black Scholes PDE solving it on a two-dimensional grid of spot prices and time")
    st.write("")
    st.write("In this app, users might try with various option parameters and select different FDM solvers, including Euler Explicit, Implicit, Crank-Nicolson, and 4th-order Runge-Kutta method.")
    st.write("")
    st.write("The project is implemented in Python and available in my github, we leverage the use of base classes to make it more readable")

    # Sidebar title and author info
    st.sidebar.title("Finite difference pricing App 📈")
    st.sidebar.markdown("**Made by:**")
    st.sidebar.markdown("[![LinkedIn](https://img.icons8.com/fluent/30/000000/linkedin.png) Juan Francisco Perez ](https://www.linkedin.com/in/juand2/)", unsafe_allow_html=True)
    st.sidebar.markdown("[![GitHub](https://img.icons8.com/fluent/30/000000/github.png) github.com/johnfcoltrane347 ](https://github.com/johnfcoltrane347) ", unsafe_allow_html=True)

    st.sidebar.write("")  # spacing for visuals
    st.sidebar.header("FDM Parameters")
    N = st.sidebar.number_input("Number of Time Steps (N)", min_value=1, value=100)
    Nj = st.sidebar.number_input("Number of Asset Price Steps (Nj)", min_value=1, value=500)
    fdm_method = st.sidebar.selectbox("FDM Method", ['Explicit', 'Implicit', 'Crank-Nicolson', 'Runge-Kutta'])

    st.sidebar.write("")  
    st.sidebar.header("Option Parameters")
    S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=90.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.0, value=1.0)
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.0, value=0.2)
    r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
    q = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, value=0.02)
    option_type = st.sidebar.selectbox("Option Type", ['Call', 'Put'])
    
    # Calculate button
    st.sidebar.write("")  
    if st.sidebar.button("Calculate 💻"):
        with st.spinner('Calculating...'):
            # Create option and PDE
            option = Option(S=S, K=K, T=T, sigma=sigma, r=r, q=q, option_type=option_type)
            pde = BlackScholesPDE(option)

            # Choose FDM solver based on user selection
            if fdm_method == 'Explicit':
                fdm = ExplicitFDM(pde, N=N, Nj=Nj)
                coeffs = r"""
                \begin{align*}
                pu &= 0.5 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + \frac{\nu}{\Delta x}\right) \\
                pm &= 1.0 - \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + r\right) \\
                pd &= 0.5 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 - \frac{\nu}{\Delta x}\right)
                \end{align*}
                """
                formula = r"""
                V_{i,j+1} = pd \cdot V_{i-1,j} + pm \cdot V_{i,j} + pu \cdot V_{i+1,j}
                """
            elif fdm_method == 'Implicit':
                fdm = ImplicitFDM(pde, N=N, Nj=Nj)
                coeffs = r"""
                \begin{align*}
                pu &= -0.5 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + \frac{\nu}{\Delta x}\right) \\
                pm &= 1.0 + \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + r\right) \\
                pd &= -0.5 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 - \frac{\nu}{\Delta x}\right)
                \end{align*}
                """
                formula = r"""
                -pu \cdot V_{i-1,j+1} + pm \cdot V_{i,j+1} - pd \cdot V_{i+1,j+1} = V_{i,j}
                """
            elif fdm_method == 'Crank-Nicolson':
                fdm = Crank_NicolsonFDM(pde, N=N, Nj=Nj)
                coeffs = r"""
                \begin{align*}
                pu &= -0.25 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + \frac{\nu}{\Delta x}\right) \\
                pm &= 1.0 + 0.5 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 + r\right) \\
                pd &= -0.25 \cdot \Delta t \cdot \left(\left(\frac{\sigma}{\Delta x}\right)^2 - \frac{\nu}{\Delta x}\right)
                \end{align*}
                """
                formula = r"""
                -pu \cdot V_{i-1,j+1} + (1+pm) \cdot V_{i,j+1} - pd \cdot V_{i+1,j+1} = pu \cdot V_{i-1,j} + (1 - pm) \cdot V_{i,j} + pd \cdot V_{i+1,j}

                """
            elif fdm_method == 'Runge-Kutta':
                fdm = RungeKuttaFDM(pde, N=N, Nj=Nj)
                coeffs = r"""
                \begin{align*}
                k_1 &= \Delta t \cdot F(V_i, S_i, t) \\
                k_2 &= \Delta t \cdot F\left(V_i + \frac{1}{2}k_1, S_i, t + \frac{1}{2}\Delta t\right) \\
                k_3 &= \Delta t \cdot F\left(V_i + \frac{1}{2}k_2, S_i, t + \frac{1}{2}\Delta t\right) \\
                k_4 &= \Delta t \cdot F(V_i + k_3, S_i, t + \Delta t) \\
                \end{align*}

                """
                formula = r"""
                V_{i,j+1} = V_{i,j} + \frac{1}{6} \left(k_1 + 2k_2 + 2k_3 + k_4\right)
                """
            else:
                fdm = None
                formula = ""

            st.header(f" {fdm_method} Method Formula")
            st.latex(formula)
            st.header(f" ")


            # Compute option price and analytical price
            price = fdm.solve()
            analytical_price = option.black_scholes()

            # Display option prices and convergence analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Option Pricing Results")
                st.markdown(f"#### Calculated Option Price")
                st.markdown(f"<div style='background-color:blue;padding:20px;font-size:30px;color:white;font-weight:bold;text-align:center'>{price:.6f}</div>", unsafe_allow_html=True)
                st.markdown(f"#### Analytical Option Price")
                st.markdown(f"<div style='background-color:green;padding:20px;font-size:30px;color:white;font-weight:bold;text-align:center'>{analytical_price:.6f}</div>", unsafe_allow_html=True)
                
                st.write("") 
                
                # Plot convergence analysis
                N_values = [50, 100, 200, 300, 500, 800, 1000, 2000]
                convergence_prices = []

                # Iterate over N values to compute convergence
                for n in N_values:
                    if fdm_method == 'Explicit':
                        fdm = ExplicitFDM(pde, N=n, Nj=Nj)
                    elif fdm_method == 'Implicit':
                        fdm = ImplicitFDM(pde, N=n, Nj=Nj)
                    elif fdm_method == 'Crank-Nicolson':
                        fdm = Crank_NicolsonFDM(pde, N=n, Nj=Nj)
                    elif fdm_method == 'Runge-Kutta':
                        fdm = RungeKuttaFDM(pde, N=n, Nj=Nj)

                    price = fdm.solve()
                    convergence_prices.append(price)
                
            with col2:
                st.markdown("### FDM Method Details")
                st.markdown(f"#### {fdm_method} Method")
                st.markdown("##### Coefficients:")
                st.latex(coeffs)

        #Plot convergence analysis
        st.header("Convergence Analysis")
        plot_convergence_analysis(N_values, convergence_prices, analytical_price)
                
if __name__ == '__main__':
    main()
