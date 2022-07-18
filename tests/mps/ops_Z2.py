import yast
import yamps
try:
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_Z2, config_Z2_fermionic


def random_seed(seed):
    config_Z2.backend.random_seed(seed)


def mps_random(N=2, Dblock=2, total_parity=0, dtype='float64'):
    psi = yamps.Mps(N)
    tc = (0, 1)
    Dc = (1, 1)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config_Z2, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi


def mpo_random(N=2, Dblock=2, total_parity=0, t_out=None, t_in=(0, 1)):
    psi = yamps.Mpo(N)
    if t_out is None:
        t_out = t_in
    Din = (1,) * len(t_in)
    Dout = (1,) * len(t_out)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config_Z2, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi


def mpo_XX_model(N, t, mu):
    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config_Z2, s=[1, 1, -1, -1], n=0)
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[t, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, t], Ds=(1, 1, 1, 2))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[0, 1], Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[[0, 0], [t, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[[0, 0], [0, t]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[[1, 0], [0, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[[0, 0], [1, 0]], Ds=(2, 1, 1, 2))
    return H


def mpo_occupation(N):
    gen = yamps.GenerateOpEnv(N, config=config_Z2)
    gen.use_default()
    H_str = "\sum_{j=0}^{"+str(N-1)+"} cp_{j}.c_{j}"
    H = gen.latex2yamps(H_str)
    return H


def mpo_gen_XX(chain, t, mu):
    gen = yamps.GenerateOpEnv(N=chain, config=config_Z2_fermionic)
    gen.use_default()
    parameters = {"t": t, "mu": mu}
    H_str = "\sum_{j=0}^{"+str(chain-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(chain-2)+"} cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(chain-2)+"} t*cp_{j+1}.c_{j}"
    H = gen.latex2yamps(H_str, parameters)
    return H


def mpo_Ising_model(N, Jij, gi):
    """ 
    MPO for Hamiltonian sum_i>j Jij Zi Zj + sum_i Jii Zi - sum_i gi Xi.
    For now only nearest neighbour coupling -- # TODO make it general
    """
    gen = yamps.GenerateOpEnv(N, config=config_Z2)
    gen.use_default(basis_type='pauli_matrices')
    parameters = {"J": Jij, "g": -gi}
    H_str = "\sum_{j=0}^{"+str(N-1)+"} g*x_{j} +\sum_{j=0}^{"+str(N-1)+"} J*z_{j} + \sum_{j=0}^{"+str(N-2)+"} J*z_{j}.z_{j+1}"
    H = gen.latex2yamps(H_str, parameters)
    return H
