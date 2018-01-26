import numpy as np
import Utils
import MDPSimulator
import MDPSolver
import math

class InvestmentGame():
    def __init__(self,**kwargs):
        self.p_noise = kwargs.get("p_noise",0.1)
        self.mean_state1 = kwargs.get("mean_state1",2)
        self.std_state1 = kwargs.get("std_state1",math.sqrt(2))
        self.P = np.array([ [[1-self.p_noise, self.p_noise],[1-self.p_noise,self.p_noise]], [[self.p_noise,1-self.p_noise],[self.p_noise, 1-self.p_noise]]])
        self.R = np.array([1,self.mean_state1])
        self.R_std = np.array([0,self.std_state1])
        self.x = 0

    def step(self, action):
        nxt_x = np.random.choice(list(range(2)),p=self.P[action,self.x])
        r_nxt = self.R[nxt_x] + self.R_std[nxt_x] * np.random.normal()

        return nxt_x, r_nxt

    def compute_J_M_V(self, mu0, gamma):
        P = self.P[0]*mu0 + self.P[1]*(1-mu0)
        J = MDPSolver.get_J(P, self.R,gamma)
        #J = np.dot(np.linalg.inv(np.identity(2) - gamma * P), self.R_mean)
        #r_m = self.R_mean*self.R_mean + (2*gamma * self.R_mean) * np.dot(P,J)
        return J


def exp2():
    '''
    Doing the deep experiment
    '''
    return

#exp2()

def exp1(gamma=0.1, mu0 = 0.5, alpha = 10, eps_greedy = 0.0, p_noise=0):
    game = InvestmentGame(p_noise=p_noise)
    mdp = MDPSimulator.generate_investment_sim(p_noise=p_noise)
    policy = [mu0, 1-mu0]
    J1 = np.zeros(2)
    nxt_x, r_nxt = 0,0
    T = 10000
    J1_vec = np.zeros((T,2))

    for k in range(T):
        x,r = nxt_x, r_nxt
        if np.random.uniform(0,1)<eps_greedy:
            action = np.random.choice(2)
        else:
            action = np.random.choice(range(2), p=policy)
        nxt_x, r_nxt = game.step(action)
        if k>T-100:
            print("action = {}, nxt_x={}, r_nxt={}".format(action, nxt_x, r_nxt))
        J1[x] = J1[x] + alpha/(k+1) * (r + gamma * J1[nxt_x] - J1[x])
        J1_vec[k, :] = J1


    J0 = game.compute_J_M_V(mu0, gamma)
    P,R = MDPSolver.get_MRP(mdp=mdp,mu=policy)
    J2 = MDPSolver.get_J(P,R,gamma)

    print("J0={}".format(J0))
    print("J2={}".format(J0))
    print("J1={}".format(J1))

    import matplotlib.pyplot as plt
    plt.plot(J1_vec)
    plt.ylabel('some numbers')
    plt.show()


exp1()