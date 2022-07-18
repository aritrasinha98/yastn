from ast import operator
from turtle import position
import numpy as np
import yast
import yamps
import re
try:
    from .configs import config_dense, config_dense_fermionic
except ImportError:
    from configs import config_dense, config_dense_fermionic


def random_seed(seed):
    config_dense.backend.random_seed(seed)


def mps_random_fermionic(N=2, Dmax=2, d=2, dtype='float64'):
    if isinstance(d, int):
        d = [d]
    d *= (N + len(d) - 1) // len(d)

    psi = yamps.Mps(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config_dense_fermionic, s=(1, 1, -1), D=[Dl, d[n], Dr], dtype=dtype)
    return psi

def mps_random(N=2, Dmax=2, d=2, dtype='float64'):
    if isinstance(d, int):
        d = [d]
    d *= (N + len(d) - 1) // len(d)

    psi = yamps.Mps(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config_dense, s=(1, 1, -1), D=[Dl, d[n], Dr], dtype=dtype)
    return psi


def mpo_random(N=2, Dmax=2, d_out=None, d=2):
    if d_out is None:
        d_out = d
    if isinstance(d, int):
        d = [d]
    d *= ((N + len(d) - 1) // len(d))
    if isinstance(d_out, int):
        d_out = [d_out]
    d_out *= ((N + len(d_out) - 1) // len(d_out))

    psi = yamps.Mpo(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config_dense, s=(1, 1, -1, -1), D=[Dl, d_out[n], d[n], Dr])
    return psi


def mpo_XX_model(N, t, mu):
    cp = np.array([[0, 0], [1, 0]])
    c = np.array([[0, 1], [0, 0]])
    nn = np.array([[0, 0], [0, 1]])
    ee = np.array([[1, 0], [0, 1]])
    oo = np.array([[0, 0], [0, 0]])

    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):  # empty tensors
        H.A[n] = yast.Tensor(config=config_dense, s=(1, 1, -1, -1))
        if n == H.first:
            tmp = np.block([[mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((1, 2, 4, 2))
            Ds = (1, 2, 2, 4)
        elif n == H.last:
            tmp = np.block([[ee], [c], [cp], [mu * nn]])
            tmp = tmp.reshape((4, 2, 1, 2))
            Ds = (4, 2, 2, 1)
        else:
            tmp = np.block([[ee, oo, oo, oo],
                            [c, oo, oo, oo],
                            [cp, oo, oo, oo],
                            [mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((4, 2, 4, 2))
            Ds = (4, 2, 2, 4)
        tmp = np.transpose(tmp, (0, 1, 3, 2))
        H.A[n].set_block(val=tmp, Ds=Ds)
    return H


def mpo_occupation(N):
    nn = np.array([[0, 0], [0, 1]])
    ee = np.array([[1, 0], [0, 1]])
    oo = np.array([[0, 0], [0, 0]])

    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):  # empty tensors
        H.A[n] = yast.Tensor(config=config_dense, s=(1, 1, -1, -1))
        if n == H.first:
            tmp = np.block([[nn, ee]])
            tmp = tmp.reshape((1, 2, 2, 2))
            Ds = (1, 2, 2, 2)
        elif n == H.last:
            tmp = np.block([[ee], [nn]])
            tmp = tmp.reshape((2, 2, 1, 2))
            Ds = (2, 2, 2, 1)
        else:
            tmp = np.block([[ee, oo],
                            [nn, ee]])
            tmp = tmp.reshape((2, 2, 2, 2))
            Ds = (2, 2, 2, 2)
        tmp = np.transpose(tmp, (0, 1, 3, 2))
        H.A[n].set_block(val=tmp, Ds=Ds)
    return H


def mpo_gen_XX(chain, t, mu):
    # prepare generator
    gen = yamps.GenerateOpEnv(N=chain, config=config_dense_fermionic)
    gen.use_default()
    ## generate directly, most general
    #H = gen.sum([yamps.mpo_term(mu, (n, n), ('cp', 'c')) for n in range(gen.N)]) \
    #    + gen.sum([yamps.mpo_term(t, (n, n+1), ('cp', 'c')) for n in range(gen.N-1)]) \
    #    + gen.sum([yamps.mpo_term(t, (n+1, n), ('cp', 'c')) for n in range(gen.N-1)])
    # generate Mpo from string, user firiendly but not very good
    parameters = {"t": t, "mu": mu}
    H_str = "\sum_{j=0}^{"+str(chain-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(chain-2)+"} cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(chain-2)+"} t*cp_{j+1}.c_{j}"
    H = gen.latex2yamps(H_str, parameters)
    return H


def mpo_Ising_model_old(N, Jij, gi):
    """ 
    MPO for Hamiltonian sum_i>j Jij Zi Zj + sum_i Jii Zi - sum_i gi Xi.
    For now only nearest neighbour coupling -- # TODO make it general
    """
    Ds, s = (2, 2), (1, -1)

    X = yast.Tensor(config=config_dense, s=s)
    X.set_block(Ds=Ds, val=[[0, 1], [1, 0]])

    Z = yast.Tensor(config=config_dense, s=s)
    Z.set_block(Ds=Ds, val=[[1, 0], [0, -1]])

    I = yast.Tensor(config=config_dense, s=s)
    I.set_block(Ds=Ds, val=[[1, 0], [0, 1]])

    X = X.add_leg(axis=-1, s=-1).add_leg(axis=0, s=1)
    Z = Z.add_leg(axis=-1, s=-1).add_leg(axis=0, s=1)
    I = I.add_leg(axis=-1, s=-1).add_leg(axis=0, s=1)

    H = yamps.Mpo(N)

    for n in H.sweep(to='last'):  # empty tensors
        if n == H.first:
            H.A[n] = yast.block({(2, 0): gi * X, (2, 1): Jij * Z, (2, 2) : I},
                                common_legs=(1, 2))
        elif n == H.last:
            H.A[n] = yast.block({(0, 0): I,
                                 (1, 0): Z,
                                 (2, 0): gi * X, (2, 1): Jij * Z, (2, 2) : I},
                                common_legs=(1, 2))
        else:
            H.A[n] = yast.block({(0, 0): I,
                                 (1, 0): Z,
                                 (2, 0): gi * X},
                                common_legs=(1, 2))
    return H


def mpo_Ising_model(N, Jij, gi):
    """ 
    MPO for Hamiltonian sum_i>j Jij Zi Zj + sum_i Jii Zi - sum_i gi Xi.
    For now only nearest neighbour coupling -- # TODO make it general
    """
    # prepare generator
    gen = yamps.GenerateOpEnv(N, config=config_dense)
    gen.use_default(basis_type='pauli_matrices')
    # generate Mpo from string, user firiendly but not very good
    parameters = {"J": Jij, "g": -gi}
    H_str = "\sum_{j=0}^{"+str(N-1)+"} g*x_{j} +\sum_{j=0}^{"+str(N-1)+"} J*z_{j} + \sum_{j=0}^{"+str(N-2)+"} J*z_{j}.z_{j+1}"
    H = gen.latex2yamps(H_str, parameters)
    return H
