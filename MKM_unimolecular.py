# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:19:46 2021

@author: cvr52
"""
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

#
# Python script to solve unimolecular reactions
# Section 4.1 - Unimolecular reaction
#

def main():
    
    for i in range(1,11):
        #i = i/10 + 1 #uncomment for smaller changes to total site number.
        x,y = solve_odes(1200,i) 
        plt.plot(x, y[:,3], label='1/'+str(i) +'Sites')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('partial pressure reactant A [atm]')
    plt.show()
    
    if 0:
        plt.figure()
        labels = ['A','B','*']
        for i in range(0, len(labels)):
            plt.semilogx(x, y[:,i], label=labels[i])
        plt.legend()
        plt.show()
        
    if 0:
        plt.figure()
        labels = ['P_A','P_B']
        for i in range(0, len(labels)):
            plt.semilogx(x, y[:,i+3], label=labels[i])
        plt.legend()
        plt.show()
        
    if 0:
        #plt.figure()
        labels = ['P_A']
        for i in range(0, len(labels)):
            plt.semilogx(x, y[:,i+3], label=labels[i])
        plt.legend()
        plt.xlabel('log time [s]')
        plt.ylabel('partial pressure [atm]')
        plt.show()
                


def solve_odes(T,site_frac):
    # initial conditions
    t0 = 0
    t1 = 1.0e-5   # total integration time
    T = 1200    # temperature in K
    #pa = 101325    # pressure of A in Pa
    #pb = 0      # pressure of B in Pa
    y0 = [0,0,1/site_frac,1,0] #[O_A,O_B,O_V,P_A,P_B]

    # construct ODE solver
    r = ode(dydt).set_integrator('vode', method='bdf',
           atol=1e-8, rtol=1e-8, nsteps=1000, with_jacobian=True)
    r.set_initial_value(y0, t0).set_f_params([T])
    
    # integrate on a logaritmic scale
    xx = np.linspace(-12.0, np.log10(t1), int((np.log10(t1) + 12.0) * 10))
    yy = []
    tt = []
    for x in xx:
        tnew = 10.0**x
        tt.append(tnew)
        yy.append(r.integrate(tnew))
        
    return tt, np.matrix(yy)


def dydt(t, y, params):
    """
    Set of ordinary differential equations
    """
    T =  params[0]    

    dydt = np.zeros(5)
    
    ma = 1.66054e-27
    mb = 1.66054e-27
    
    k_ads_a = calc_kads(T, y[3]*101325, 1e-20, ma)
    k_des_a = calc_kdes(T, 1e-20, ma, 1, 1, 120e3)
    
    k_ads_b = calc_kads(T, y[4]*101325, 1e-20, mb)
    k_des_b = calc_kdes(T, 1e-20, mb, 1, 1, 220e3)
    
    kf = calc_k_arr(T, 1e13, 50e3)
    kb = calc_k_arr(T, 1e13, 150e3)
    
    # Note: The partial pressures for the rates of ads/des are implicitly 
    # calculated within k_ads_a/k_des_a and k_ads_b/k_des_b. They were 
    # based on collision theory.
    dydt[0] = k_ads_a * y[2] - k_des_a * y[0] - kf * y[0] + kb * y[1]
    dydt[1] = k_ads_b * y[2] - k_des_b * y[1] + kf * y[0] - kb * y[1]
    dydt[2] = -k_ads_a * y[2] + k_des_a * y[0] - k_ads_b * y[2] + k_des_b * y[1]
    
    dydt[3] = -(k_ads_a * y[2] - k_des_a * y[0])
    dydt[4] = -(k_ads_b * y[2] - k_des_b * y[1])
    
    
    return dydt

def calc_k_arr(T, nu, Eact):
    """
    Calculate reaction rate constant for a surface reaction
    
    T       - Temperature in K
    nu      - Pre-exponential factor in s^-1
    Eact    - Activation energy in J/mol
    """
    R = 8.3144598 # gas constant
    return nu * np.exp(-Eact / (R * T))

def calc_kads(T, P, A, m):
    """
    Reaction rate constant for adsorption
    
    T           - Temperature in K
    P           - Pressure in Pa
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    """
    kb = 1.38064852E-23 # boltzmann constant
    return P*A / np.sqrt(2 * np.pi * m * kb * T)

def calc_kdes(T, A, m, sigma, theta_rot, Edes):
    """
    Reaction rate constant for desorption
    
    T           - Temperature in K
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    sigma       - Symmetry number
    theta_rot   - Rotational temperature in K
    Edes        - Desorption energy in J/mol
    """
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    R = 8.3144598       # gas constant
    return kb * T**3 / h**3 * A * (2 * np.pi * m * kb) / \
        (sigma * theta_rot) * np.exp(-Edes / (R*T))

if __name__ == '__main__':
    main()