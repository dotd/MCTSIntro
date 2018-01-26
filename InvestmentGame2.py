import MDPSimulator
import MDPSolver
import numpy as np
import copy
import time

mdp = MDPSimulator.generate_investment_sim()
mu = MDPSimulator.generate_uniform_policy(mdp.X, mdp.U)
gamma = 0.5
num_samples = 10000

P, R, R_std = MDPSolver.get_MRP(mdp, mu)

start = time.time()
J_exact = MDPSolver.get_J(P, R, gamma)
print("J_exact={}".format(J_exact))
print(time.time() - start)

R_M2 = MDPSimulator.get_R_M2(P, R, R_std, gamma, J_exact)
start = time.time()
M2_exact = MDPSolver.get_J(P, R_M2, gamma**2)
print("M2_exact={}".format(M2_exact))
print(time.time() - start)

trajectory = mdp.simulate(0, mu, num_samples=num_samples)
start = time.time()
J_td = MDPSolver.get_J_as_TD(trajectory=trajectory, gamma=gamma, X=mdp.X, alpha=10)
print("J_td={}".format(J_td))
print(time.time() - start)

if num_samples<=1001:
    start = time.time()
    J_MC = MDPSolver.get_J_as_MC(trajectory, gamma, X=mdp.X)
    print("J_MC={}".format(J_MC))
    print(time.time() - start)

    start = time.time()
    M2_MC = MDPSolver.get_J_as_MC(trajectory, gamma, X=mdp.X, func=lambda x:x*x)
    print("M2_MC={}".format(M2_MC))
    print(time.time() - start)
else:
    print("Skip straight forward MonteCarlo simulation")

start = time.time()
J_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X)
print("J_MC_filt={}".format(J_MC_filt))
print(time.time() - start)

start = time.time()
M2_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X, func=lambda x:x*x)
print("M2_MC_filt={}".format(M2_MC_filt))
print(time.time() - start)

start = time.time()
phi = np.random.normal(size=(mdp.X,mdp.X))
w_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma, r=R)
J_exact_LSTD = np.dot(phi, w_exact_LSTD)
print("J_exact_LSTD={}".format(J_exact_LSTD))
print(time.time() - start)
