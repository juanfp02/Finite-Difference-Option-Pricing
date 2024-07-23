import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Option_Base import Option
from FDM_base import FDMBase
from BlackScholesPDE import BlackScholesPDE
from FDM_solvers import ExplicitFDM, ImplicitFDM, Crank_NicolsonFDM, RungeKuttaFDM
from helper_Functions import plot_surface, plot_convergence_analysis

def main():
    st.title("Option Pricing with Finite Difference Methods")

    st.write("In this application, we explore 4 different FDM Methods applied to the pricing of European options."
             "Please check out my github repo for the code implementation")


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
    if st.sidebar.button("Calculate"):
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









