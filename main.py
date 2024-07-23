from Option_Base import Option
from FDM_base import FDMBase
from BlackScholesPDE import BlackScholesPDE
from FDM_solvers import ExplicitFDM, ImplicitFDM, Crank_NicolsonFDM, RungeKuttaFDM
import matplotlib.pyplot as plt
import numpy as np


"""

Example usage of FDM in option pricing


"""


def main():

    # Define Option:
    option = Option(S=90, K=100, T=1, sigma=0.2, r=0.05, q=0.02, option_type='Call')

    # Create PDE
    BSPDE = BlackScholesPDE(option)

    # Create a FDM derived object
    fdm = ExplicitFDM(BSPDE, N=100, Nj=500)

    # Call the solve method to return price
    price = fdm.solve()

    # Black Scholes Price to compare result
    analytical_price = option.black_scholes()


    # Convergence analysis

    N = [50,100,200,500,1000,2000]

    for i in N:
        

        fdm = ExplicitFDM(BSPDE, N=i, Nj=500)
        price = fdm.solve()

        print(f"The option price is: {price}")
        print(f"analytical price {price}")




if __name__ == '__main__':
    main()