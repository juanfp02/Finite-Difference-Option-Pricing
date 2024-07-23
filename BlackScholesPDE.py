
"""

Black Scholes PDE
JFPerez
July '24

"""


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







"""

Black Scholes PDE
JFPerez
July '24

"""