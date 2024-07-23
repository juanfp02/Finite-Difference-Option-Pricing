from Option_Base import Option
from FDM_base import FDMBase
from BlackScholesPDE import BlackScholesPDE
from FDM_solvers import ExplicitFDM, ImplicitFDM, Crank_NicolsonFDM, RungeKuttaFDM
from helper_Functions import plot_surface
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Example usage:
    option = Option(S=90, K=100, T=1, sigma=0.2, r=0.05, q=0.02, option_type='Call')
    BSPDE = BlackScholesPDE(option)
    fdm = ExplicitFDM(BSPDE, N=100, Nj=500)
    price = fdm.solve()
    analytical_price = option.black_scholes()


    N = [50,100,200,500,1000,2000]

    for i in N:
        

        fdm = ExplicitFDM(BSPDE, N=i, Nj=500)
        price = fdm.solve()

        print(f"The option price is: {price}")
        print(f"analytical price {price}")

    S_mesh, T_Mesh, C_Mesh = fdm.Surface_data()

    plot_surface(S_mesh,T_Mesh,C_Mesh)




if __name__ == '__main__':
    main()