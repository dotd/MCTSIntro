import numpy as np
from MDPSimulator import MDPSim
import MDPSimulator

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
            for u in range(4):

                x_nxt = x
                y_nxt = y
                # 0->up
                if u==0:
                    if y>0 and grid[y-1,x] is not None:
                        P[u,two2one(x,y),two2one(x,y-1)] = 1
                    else:
                        P[u, two2one(x, y), two2one(x, y )] = 1
                # 1->right
                if u==1:
                    if x<size_x-1 and grid[y,x+1] is not None:
                        P[u,two2one(x,y),two2one(x+1,y)] = 1
                    else:
                        P[u, two2one(x, y), two2one(x, y )] = 1
                # 2->down
                if u==2:
                    if y<size_y-1 and grid[y+1,x] is not None:
                        P[u,two2one(x,y),two2one(x,y+1)] = 1
                    else:
                        P[u, two2one(x, y), two2one(x, y )] = 1
                # 3->left
                if u==3:
                    if x>0 and grid[y,x-1] is not None:
                        P[u, two2one(x, y), two2one(x-1, y)] = 1
                    else:
                        P[u, two2one(x, y), two2one(x, y)] = 1

                if grid[y,x] is not None:
                    R[u,x,y] = grid[y,x]


    # normalizing the MDP
    for u in range(U):
        for x in range(X):
            P[u,x] = P[u,x] / np.sum(P[u,x])
    info = {"size_x":size_x, "size_y":size_y}
    return MDPSim(P,R,R_std, info=info)

def show_policy(mdp,mu):
    size_x = mdp.info["size_x"]
    size_y = mdp.info["size_y"]
    m = [["x" for x in range(size_x)] for y in range(size_y)]
    one2two = lambda k: (k % size_x, int(k / size_x))
    mu_maximal = np.argmax(mu, axis=1)
    for state in range(mdp.X):
        x,y = one2two(state)
        m[y][x] = str(mu_maximal[state])
    lines = ["".join(m[i]) for i in range(size_y)]
    s = "\n".join(lines)
    return s


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

GridWorld2DTest()