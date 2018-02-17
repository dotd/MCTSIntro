import numpy as np
from MDPSimulator import MDPSim
import MDPSimulator
import MDPSolver

def createGridWorld2D(size_x,
                      size_y,
                      noise_rate,
                      num_reward_places,
                      num_obstacles,
                      random=np.random.RandomState(0)):
    X = size_x * size_y
    U = 4;
    two2one = lambda x,y: x + y*size_x
    one2two = lambda k: (k%size_x , k/size_x)

    grid = np.zeros(shape=(size_y,size_x))
    for r in range(num_reward_places):
        xr = random.randint(0,size_x)
        yr = random.randint(0,size_y)
        grid[yr,xr] = 1
    for o in range(num_obstacles):
        xr = random.randint(0,size_x)
        yr = random.randint(0,size_y)
        grid[yr,xr] = None

    P = np.zeros(shape=(U,X,X))
    R = np.zeros(shape=(U,X,X))
    R_std = np.zeros(shape=(U,X,X))
    for u in range(U):
        P[u,:,:] = np.identity(X)*noise_rate

    for x in range(size_x):
        for y in range(size_y):
            from_idx = two2one(x,y)
            if not np.isnan(grid[y, x]):
                tmp_R = grid[y,x]
            else:
                tmp_R = 0
            for u in range(4):

                x_nxt = x
                y_nxt = y
                # 0->up
                if u==0:
                    if y>0 and not np.isnan(grid[y-1,x]):
                        P[u,from_idx,two2one(x,y-1)] = 1
                        R[u, from_idx, two2one(x, y - 1)] = tmp_R
                    else:
                        P[u, from_idx, from_idx] = 1
                        R[u, from_idx, from_idx] = tmp_R
                # 1->right
                if u==1:
                    if x<size_x-1 and not np.isnan(grid[y,x+1]):
                        P[u,from_idx,two2one(x+1,y)] = 1
                        R[u, from_idx, two2one(x+1, y)] = tmp_R
                    else:
                        P[u, from_idx, from_idx] = 1
                        R[u, from_idx, from_idx] = tmp_R
                # 2->down
                if u==2:
                    if y<size_y-1 and not np.isnan(grid[y+1,x]):
                        P[u,from_idx,two2one(x,y+1)] = 1
                        R[u, from_idx, two2one(x, y + 1)] = tmp_R
                    else:
                        P[u, from_idx, from_idx] = 1
                        R[u, from_idx, from_idx] = tmp_R
                # 3->left
                if u==3:
                    if x>0 and not np.isnan(grid[y,x-1]):
                        P[u, from_idx, two2one(x-1, y)] = 1
                        R[u, from_idx, two2one(x-1, y)] = tmp_R
                    else:
                        P[u, from_idx, from_idx] = 1
                        R[u, from_idx, from_idx] = tmp_R


    # normalizing the MDP
    for u in range(U):
        for x in range(X):
            P[u,x] = P[u,x] / np.sum(P[u,x])
    info = {"size_x":size_x, "size_y":size_y, "grid":grid}
    return MDPSim(P, R, R_std, info=info)

def show_policy(mdp,mu):
    grid = mdp.info["grid"]
    size_x = mdp.info["size_x"]
    size_y = mdp.info["size_y"]
    m = [["0" for x in range(size_x)] for y in range(size_y)]
    one2two = lambda k: (k % size_x, int(k / size_x))
    mu_maximal = np.argmax(mu, axis=1)
    for state in range(mdp.X):
        x,y = one2two(state)
        if np.isnan(grid[y, x]):
            m[y][x] = "x"
        elif grid[y, x] != 0:
            m[y][x] = "R"
        else:
            m[y][x] = str(mu_maximal[state])
    lines = ["".join(m[i]) for i in range(size_y)]
    s = "\n".join(lines)
    return s

def show_grid(mdp):
    grid = mdp.info["grid"]
    size_x = mdp.info["size_x"]
    size_y = mdp.info["size_y"]
    lines = []
    for y in range(size_y):
        line = []
        for x in range(size_x):
            if np.isnan(grid[y,x]):
                line.append("x")
            elif grid[y,x]!=0:
                line.append("R")
            else:
                line.append("0")
        lines.append("".join(line))
    return "\n".join(lines)


def show_policy_and_mdp(mdp, mu):
    pass


def GridWorld2DTest():
    size_x = 2
    size_y = 3
    noise_rate = 0
    num_reward_places = 1
    num_obstacles = 0
    mdp = createGridWorld2D(size_x=size_x, size_y=size_y, noise_rate=noise_rate,
                            num_reward_places=num_reward_places, num_obstacles=num_obstacles)
    mu = MDPSimulator.generate_deterministic_policy(mdp.X,mdp.U)
    print(show_policy(mdp,mu))
    #print(mdp.show())

def GridWorld2DTest_2():
    size_x = 20
    size_y = 15
    noise_rate = 0
    num_reward_places = 1
    num_obstacles = 0
    '''
    mdp = createGridWorld2D(size_x=size_x, size_y=size_y, noise_rate=noise_rate,
                            num_reward_places=num_reward_places, num_obstacles=num_obstacles)
    print(show_grid(mdp))
    print("\n")
    '''
    num_obstacles = 3
    mdp = createGridWorld2D(size_x=size_x, size_y=size_y, noise_rate=noise_rate,
                            num_reward_places=num_reward_places, num_obstacles=num_obstacles)
    mu = MDPSimulator.generate_deterministic_policy(mdp.X,mdp.U)
    print(show_policy(mdp,mu))

    print("\n")
    mu_PI, J_PI, Q_PI, iter_counter = MDPSolver.PI(mdp, 0.45)
    print(show_policy(mdp,mu_PI))


GridWorld2DTest_2()