import numpy as np
import math


class MDPSim:
    '''
    P, R, R_var are variance classes of sizes U*X*X
    '''
    def __init__(self, P, R, R_std = 0, random = np.random.RandomState(0)):
        self.P = P
        self.R = R
        self.R_std = R_std
        self.U = self.P.shape[0]
        self.X = self.P.shape[1]
        self.rand = random

    def simulate(self, x, policy, num_samples):
        '''
        policy of size X*U

        '''
        mu = policy[x]
        trajectory = []
        for n in range(num_samples):
            u = self.rand.choice(range(self.U), p=policy[x])
            y = self.rand.choice(range(self.U), p=self.P[u,x])
            r = self.R[u, x, y] + self.rand.normal()*self.R_std[u,x,y]
            trajectory.append([x,u,r])
            x = y
        return trajectory

    def show(self):
        lines = []
        lines.append("P=")
        lines.append(show_3dMat(self.P))
        lines.append("R=")
        lines.append(show_3dMat(self.R))
        lines.append("R_std=")
        lines.append(show_3dMat(self.R_std))
        return "\n".join(lines)


def show_2dMat(mat, **kwargs):
    sep = kwargs.get("sep","\t")
    line_sep = kwargs.get("line_sep","\n")
    lines = []
    for y in range(mat.shape[0]):
        line = []
        for x in range(mat[0].shape[0]):
            line.append(str(mat[y,x]))
        lines.append(sep.join(line))
    return line_sep.join(lines)

def show_3dMat(mat, **kwargs):
    mat_sep = kwargs.get("mat_sep","\n")
    U = mat.shape[0]
    lines = []
    for u in range(U):
        lines.append("u=" + str(u))
        lines.append(show_2dMat(mat[u],**kwargs))
    return mat_sep.join(lines)

def OneDVec2ThreeDVec(R_vec, U):
    X = R_vec.shape[0]
    R_mat = np.zeros((U,X,X))
    for x,reward in enumerate(R_vec):
        R_mat[:,x,:] = reward
    return R_mat

def ThreeDVec2OneDVec(R_mat, P, mu):
    X = R_mat.shape[1]
    U = R_mat.shape[0]

    R1D = np.zeros(shape=(X,))
    M1D = np.zeros(shape=(X,))
    std1D = np.zeros(shape=(X,))
    for x in range(X):
        for u in range(U):
            for y in range(X):
                R1D[x] += P[u,x,y] * mu[x,u] * R_mat[u,x,y]
    return R1D

def generate_investment_sim(p_noise = 0, **kwargs):
    P = np.array([[[1 - p_noise, p_noise], [1 - p_noise, p_noise]], [[p_noise, 1 - p_noise], [p_noise, 1 - p_noise]]])
    R1 = kwargs.get("R1", 2)
    R1_std = kwargs.get("R1_std", math.sqrt(2))
    R1D = np.array([1, R1])
    R1D_std = np.array([0, R1_std])
    R = OneDVec2ThreeDVec(R1D,U=2)
    R_std = OneDVec2ThreeDVec(R1D_std,U=2)
    mdp = MDPSim(P = P, R = R, R_std=R_std)
    return mdp

def generate_uniform_policy(X,U):
    policy = np.ones(shape=(X,U))/U
    return policy

def get_R_M2(P, R, R_std, gamma, J):
    R_M2 = R*R + R_std * R_std + 2* gamma * R * (np.dot(P,J))
    return R_M2

def func1():
    mdp = generate_investment_sim()
    policy = generate_uniform_policy(mdp.X,mdp.U)
    print("the mdp:")
    print(mdp.show())
    print("the policy:")
    print(show_2dMat(policy))
    trajectory = mdp.simulate(x=0,policy=policy,num_samples=10)
    print(trajectory)



