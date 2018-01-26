import numpy as np
import MDPSimulator

def get_MRP(mdp, mu):
    P = np.zeros((mdp.X,mdp.X))
    for x in range(mdp.X):
        for u in range(mdp.U):
            P[x] += mu[x,u] * mdp.P[u,x]
    R = MDPSimulator.ThreeDVec2OneDVec(mdp.R,mdp.P, mu)
    R_std = MDPSimulator.ThreeDVec2OneDVec(mdp.R_std, mdp.P, mu)
    return P, R, R_std

def get_J(P, R, gamma):
    J = np.dot(np.linalg.inv(np.identity(P.shape[0]) - gamma * P), R)
    return J

def get_rm(P, R, gamma, J=None):
    if J is None:
        J = get_J(P,R,gamma)
    rm = R*R + (2* gamma * R) * np.dor(P,J)
    return rm

def compute_reward_to_go(trajectory, idx_start, gamma):
    R = 0
    d = 1
    for idx in (range(idx_start,len(trajectory))):
        R += trajectory[idx][2] * d
        d *= gamma
    return R

def get_J_as_MC(trajectory, gamma, X = None, func = lambda x: x):
    if X is None:
        X = max(trajectory)+1
    times = np.zeros((X,))
    values = np.zeros((X,))
    for idx in range(len(trajectory)):
        x = trajectory[idx][0]
        reward2go = compute_reward_to_go(trajectory, idx, gamma)
        values[x] += func(reward2go)
        times[x] +=1
    J = values/times
    return J

def get_discount_factor_as_filter(gamma, filt_len):
    filt = np.zeros((filt_len,))
    filt[0] = 1
    for k in range(1,filt_len):
        filt[k] = filt[k-1]*gamma
    filt = np.flip(filt,0)
    return filt

def get_J_as_MC_filter(trajectory, gamma, X=None, filt_len = 40, func = lambda x: x):
    # The trajectory is x,u,r
    X = max(trajectory)+1 if X is None else X
    # get the reward
    r = np.array([vec[2] for vec in trajectory])
    # make the filter
    filt = get_discount_factor_as_filter(gamma, filt_len)
    res = np.convolve(r,filt)
    start_idx = filt_len-1 # 39 is the first index, meaning we removed 39 indices
    end_idx = start_idx + r.shape[0]
    res = res[start_idx:end_idx]

    # Do the stats
    times = np.zeros((X,))
    values = np.zeros((X,))
    for k in range(len(trajectory)):
        x = trajectory[k][0]
        times[x] +=1
        values[x] += func(res[k])
    J = values/times
    return J

def get_J_as_TD(trajectory, gamma, X, alpha):
    '''
    trajectory is x,u,r
    '''
    J1 = np.zeros(X)

    for k in range(len(trajectory)-1):
        x = trajectory[k][0]
        r = trajectory[k][2]
        y = trajectory[k+1][0]

        J1[x] = J1[x] + alpha/(k+1) * (r + gamma * J1[y] - J1[x])
    return J1

def get_exact_J_LSTD(phi, P, gamma, r):
    A = np.linalg.multi_dot([phi.T, np.identity(P.shape[0]) - gamma*P ,phi])
    b = np.dot(phi.T,r)
    w = np.linalg.solve(A, b)
    return w
