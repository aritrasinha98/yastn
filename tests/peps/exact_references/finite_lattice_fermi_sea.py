import numpy as np
import scipy.linalg as LA

def tb2D(Nx, Ny, t, mu):
    """ Generates 2D tight binding Hamiltonian of size Nx x Ny with given tunneling strength t and 
    chemical potential mu """
    Ham = np.zeros((Nx*Ny, Nx*Ny))

    for n in range(Nx):
        for m in range(Ny-1):
            Ham[m+Ny*n, m+1+Ny*n] = Ham[m+1+Ny*n, m+Ny*n] = -t 

    for m in range(Ny):
        for n in range(Nx-1):
            Ham[m+Ny*n, m+Ny*(n+1)] = Ham[m+Ny*(n+1), m+Ny*n] = -t

    np.fill_diagonal(Ham, -mu)

    return Ham


def calculate_correlator(x1, y1, x2, y2, W, V, Ny, beta):
    vecorth = lambda x, y, Ny : Ny*x + y
    sv1 = vecorth(x1, y1, Ny)
    sv2 = vecorth(x2, y2, Ny)

    return np.sum([V[sv1, i]*V[sv2, i]/(1+np.exp(beta*W[i])) for i in range(len(W))])


def correlator(site1, site2, Nx, Ny, t, mu, beta):
    """
    fermionic hopping correlator between sites (x1, y1) and (x2, y2)
    finite lattice size: Nx * Ny inverse temeprature beta 
    """
    x1, y1 = site1
    x2, y2 = site2
   
    HamT = tb2D(Nx, Ny, t, mu)
    W, V = LA.eig(HamT)           

    return calculate_correlator(x1, y1, x2, y2, W, V, Ny, beta)

Nx, Ny = 3, 3
t = 1
beta = 0.1
mu = 0
site1, site2 = (2, 0), (2, 1)
site3, site4 = (0, 1), (1, 1)

corr = []

# Define a list of all possible relative coordinates for nearest neighbors in a 2D grid
neighbors = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # right, down, left, up

for s1 in range(Nx):
    for s2 in range(Ny):
        for dx, dy in neighbors:
            s3, s4 = s1 + dx, s2 + dy
            # Check that the neighbor coordinates are valid (inside the lattice)
            if 0 <= s3 < Nx and 0 <= s4 < Ny:
                corr.append(correlator((s1, s2), (s3, s4), Nx, Ny, t, mu, beta))


print("mean correlation: ", np.mean(corr))

for site1, site2 in [(site1, site2), (site3, site4)]:
    print(f"Correlation for spinless fermi sea at inverse temperature beta {beta} between {site1} and {site2} is {correlator(site1, site2, Nx, Ny, t, mu, beta)}")
