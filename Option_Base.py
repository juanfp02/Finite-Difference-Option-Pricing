import numpy as np
from scipy.stats import norm




"""

Base option class
JFPerez
July '24

"""

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
