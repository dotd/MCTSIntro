import MDPSimulator
import MDPSolver
import numpy as np
import copy
import time

mdp = MDPSimulator.generate_random_MDP(X=5,U=3,B=5,R_sparse=5,std=1,random_state=np.random.RandomState(1))
if mdp.X<=5 and mdp.U<=5:
    print(mdp.show())

mu = MDPSimulator.generate_uniform_policy(mdp.X, mdp.U)
gamma = 0.2
num_samples = 1000

P, R, R_std = MDPSolver.get_MRP(mdp, mu)
##############################################################
start = time.time()
J_exact = MDPSolver.get_J(P, R, gamma)
print("We solve J by inversing a marix:")
print("J_exact={}".format(J_exact))
print("The time for that stage is ", time.time() - start,"\n")

##############################################################
start = time.time()
J_iteratively, norm, num_iters = MDPSolver.get_J_iteratively(P, R, gamma)
print("Solving J by Value Iteration:")
print("J_iteratively={}, norm={}, num_iters={}".format(J_iteratively, norm, num_iters))
print("The time for that stage is ", time.time() - start,"\n")

##############################################################
R_M2 = MDPSimulator.get_R_M2(P, R, R_std, gamma, J_exact)
start = time.time()
M2_exact = MDPSolver.get_J(P, R_M2, gamma**2)
print("We solve M by computing R_M2 (reward of M) and inversing a marix:")
print("M2_exact={}".format(M2_exact))
print("The time for that stage is ", time.time() - start,"\n")

##############################################################
# Create the trajectory
trajectory = mdp.simulate(0, mu, num_samples=num_samples)
#print(trajectory)

##############################################################
start = time.time()
J_td = MDPSolver.get_J_as_TD(trajectory=trajectory, gamma=gamma, X=mdp.X, alpha=10)
print("We solve J by TD learning method:")
print("J_td={}".format(J_td))
print("The time for that stage is ", time.time() - start,"\n")


start = time.time()
M2_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X, func=lambda x:x*x)
print("We solve M2 by MC_filt learning method:")
print("M2_MC_filt={}".format(M2_MC_filt))
print("The time for that stage is ", time.time() - start,"\n")

##############################################################
# Compute with loops the
if num_samples<=1001:
    start = time.time()
    J_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X)
    print("Solving J by MC_raw learning method:")
    print("J_MC={}".format(J_MC))
    print("The time for that stage is ", time.time() - start, "\n")

    start = time.time()
    M2_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X, func=lambda x: x * x)
    print("Solving M2 by MC_raw learning method:")
    print("M2_MC={}".format(M2_MC))
    print("The time for that stage is ", time.time() - start, "\n")
else:
    print("Skip straight forward MonteCarlo simulation\n")

start = time.time()
J_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X)
print("Solving J by MC_filter learning method:")
print("J_MC_filt={}".format(J_MC_filt))
print("The time for that stage is ", time.time() - start, "\n")

phi = np.random.normal(size=(mdp.X,mdp.X))
class PhiClass:
    def __init__(self, phi):
        self.phi = phi
    def get(self, x):
        return self.phi[x]

phi_class = PhiClass(phi)


start = time.time()
w_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma, r=R)
J_exact_LSTD = np.dot(phi, w_exact_LSTD)
print("Solving J by LSTD learning method with random base:")
print("J_exact_LSTD={}".format(J_exact_LSTD))
print("The time for that stage is ", time.time() - start, "\n")

start = time.time()
wm_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma**2, r=R_M2)
M_exact_LSTD = np.dot(phi, wm_exact_LSTD)
print("M_exact_LSTD={}".format(M_exact_LSTD))
V_exact_by_M_and_J = MDPSolver.get_V_by_J_M(J_exact, M2_exact)
print("V_exact_by_M_and_J={}".format(V_exact_by_M_and_J))
print("The time for that stage is ", time.time() - start, "\n")

start = time.time()
R_V_exact = MDPSimulator.get_R_V(P, R, R_std, gamma, J_exact)
V_exact_direct = MDPSolver.get_J(P, R_V_exact, gamma**2)
print("V_exact_direct={}".format(V_exact_direct))
print("The time for that stage is ", time.time() - start, "\n")

start = time.time()
w_sim = MDPSolver.get_simulation_J_LSTD(phi_class, trajectory, gamma)
print("Solving J by LSTD and *trajectory*:")
J_sim_LSTD = np.dot(phi_class.phi, w_sim)
print("J_sim_LSTD={}".format(J_sim_LSTD))
print("The time for that stage is ", time.time() - start, "\n")

start = time.time()
mu_PI, J_PI, Q_PI, iter_counter = MDPSolver.PI(mdp, gamma)
is_monotone = MDPSolver.check_J_collector_monotone(J_PI, debug_print=True)
print("mu_PI={}\n(final) J_PI={}\nQ_PI={}\niter_counter={}\nis_monotone={}".format(mu_PI, J_PI[-1], Q_PI, iter_counter, is_monotone))
print("The time for that stage is ", time.time() - start, "\n")

start = time.time()
J_VI, mu_VI, iter_VI, delta_VI = MDPSolver.VI(mdp, gamma, theta = 1e-12)
print("J_VI={}\nmu_VI={}\niter_VI={}\ndelta_VI={}".format(J_VI, mu_VI, iter_VI, delta_VI))
print("The time for that stage is ", time.time() - start, "\n")
