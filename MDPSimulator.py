import numpy as np
import math


class MDPSim:
    '''
    P, R, R_var are variance classes of sizes U*X*X
    '''
    def __init__(self, P, R, R_std = 0, random = np.random.RandomState(0), basis = None, info = {}):
        self.P = P
        self.R = R
        self.R_std = R_std
        self.U = self.P.shape[0]
        self.X = self.P.shape[1]
        self.rand = random
        self.cur_state = None
        self.basis = basis
        if self.basis is not None:
            self.D = self.basis.shape[1]
        self.reset()
        self.info = info

    def reset(self):
        self.cur_state = self.rand.randint(self.X)
        return self.get_cur_state()

    def simulate(self, x, policy, num_samples):
        '''
        policy of size X*U

        '''
        mu = policy[x]
        trajectory = []
        for n in range(num_samples):
            u = self.rand.choice(range(self.U), p=policy[x])
            y = self.rand.choice(range(self.X), p=self.P[u,x])
            r = self.R[u, x, y] + self.rand.normal()*self.R_std[u,x,y]
            trajectory.append([x,u,r])
            x = y
        return trajectory

    def step(self,u):
        x = self.cur_state
        y = self.rand.choice(range(self.X), p=self.P[u, x])
        r = self.R[u, x, y] + self.rand.normal()*self.R_std[u,x,y]
        self.cur_state = y
        if self.basis is None:
            # Gym format
            return y,r, None, None
        else:
            return self.basis[y],r, None, None

    def show(self):
        lines = []
        lines.append("P=")
        lines.append(show_3dMat(self.P))
        lines.append("R=")
        lines.append(show_3dMat(self.R))
        lines.append("R_std=")
        lines.append(show_3dMat(self.R_std))
        return "\n".join(lines)

    def get_cur_state(self):
        if self.basis is None:
            return self.cur_state
        else:
            return self.basis[self.cur_state]



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

def generate_deterministic_policy(X, U, random = np.random.RandomState(0)):
    policy = np.zeros(shape=(X,U))
    for x in range(X):
        u = random.randint(0,U)
        policy[x,u] = 1.0
    return policy

def get_R_M2(P, R, R_std, gamma, J):
    R_M2 = R*R + R_std * R_std + 2* gamma * R * (np.dot(P,J))
    return R_M2

def get_R_V(P, R, R_std, gamma, J):
    Jy = np.dot(P,J)
    R_V = gamma*gamma * ( np.dot(P,J*J) - Jy * Jy) + R_std*R_std
    return R_V

def func1():
    mdp = generate_investment_sim()
    policy = generate_uniform_policy(mdp.X,mdp.U)
    print("the mdp:")
    print(mdp.show())
    print("the policy:")
    print(show_2dMat(policy))
    trajectory = mdp.simulate(x=0,policy=policy,num_samples=10)
    print(trajectory)

def get_random_sparse_vector(X, B, to_normalize, type, random_state):
    vec = np.zeros(shape=(X,))
    if type=="gaussian":
        vec[0:B] = random_state.normal(size=(B,))
    else:
        vec[0:B] = random_state.uniform(low=0, high=1.0, size=(B,))

    if to_normalize:
        sum_vec = np.sum(vec[0:B])
        vec[0:B] = vec[0:B] / sum_vec

    vec = random_state.permutation(vec)
    return vec


def generate_random_MDP(X, U, B, R_sparse, std = 0, random_state = np.random.RandomState(0), basis = None):
    P = np.zeros(shape=(U,X,X))
    R = np.zeros(shape=(U,X,X))
    R_std = std*np.ones(shape=(U,X,X))

    for u in range(U):
        for x in range(X):
            P[u, x] = get_random_sparse_vector(X, B, True, "uniform", random_state)
            R[u, x] = get_random_sparse_vector(X, R_sparse, False, "gaussian", random_state)
            R[u,x,:]  = R[u,x,0]
        if u>=1:
            R[u] = R[0]

    if basis is not None:
        basis = random_state.normal(size=(X, basis))
    mdp = MDPSim(P = P, R = R, R_std=R_std, basis = basis)
    return mdp

def func2():
    mdp = generate_random_MDP(X=5,U=3,B=2,R_sparse=1)
    print(mdp.show())

