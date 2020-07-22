import numpy as np
from yamps.mps import Env3
from yamps.mps import measure
from yamps.tensor.eigs import expmw
#################################
#           tdvp                #
#################################



def tdvp_sweep_1site(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r""" 
    Perform sweep with 1site-TDVP by applying exp(-i*dt*H) on initial vector. Note the convention for time step sign.
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        initial state.
        can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        operator given in MPO decomposition. 
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    M: Mps, nr_phys=1
        Kraus operators.
        legs are [Kraus dimension, ket-physical, bra-physical]

    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
        
    dt: double
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution 

    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        is MPO hermitian

    fermionic: bool
        default = False
        use while pllying a SWAP gate. True for fermionic systems.
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    bi_orth: bool
        default=True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors. 
            
    NA: bool
        default=None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.    
        Option for exponentiation = exp(). 
            
    opts_svd: dict
        default=None
        options for truncation on virtual d.o.f.

    optsK_svd: dict
        default=None
        options for truncation on auxilliary d.o.f.

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """
    # change. adjust the sign according to the convention
    sgn = 1j * ( np.sign(dt.real) + 1j * np.sign(dt.imag) )
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()
        
    for n in psi.g.sweep(to='last'): # sweep from fist to last 
        if M: # apply the Kraus operator
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
        
        if H: # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], Bv=lambda v : env.Heff1(v, n, conj=True), dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]

        # canonize and save
        if opts_svd != None:
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), sU=-1, **opts_svd)
            psi.A[n] = U
            psi.pC = (n,n+1)
            psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        if H and n != psi.N - 1:# backward in time evolution of a central site: T(-dt*.5)
            init = psi.A[psi.pC]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff0(v, psi.pC), init=[init], Bv=lambda v : env.Heff0(v, psi.pC, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff0(v, psi.pC), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # canonize and save
            if opts_svd:
                U, S, V = init[0].split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
            else:
                psi.A[psi.pC] = init[0]
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        init = psi.A[n]
        if H: # forward in time evolution of a central site: T(+dt*.5)
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], Bv=lambda v : env.Heff1(v, n, conj=True), dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
        
        if M: # apply the Kraus operator
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            init = init.transpose(axes=(0,1,3,2))

        # canonize and save
        if opts_svd:
            U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), sU=-1, **opts_svd)
            psi.A[n] = V
            psi.pC = (n - 1,n)
            psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

        if H and n != 0: # backward in time evolution of a central site: T(-dt*.5)
            init = psi.A[psi.pC]            
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff0(v, psi.pC), init=[init], Bv=lambda v : env.Heff0(v, psi.pC, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff0(v, psi.pC), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # canonize and save
            if opts_svd != None:
                U, S, V = init[0].split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
            else:
                psi.A[psi.pC] = init[0]
        psi.absorb_central(towards=psi.g.first)
            
    return env

def tdvp_sweep_2site(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r""" 
    Perform sweep with 2site-TDVP.
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        initial state.
        can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        operator given in MPO decomposition. 
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    M: Mps, nr_phys=1
        Kraus operators.
        legs are [Kraus dimension, ket-physical, bra-physical]

    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
        
    dt: double
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution 

    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        is MPO hermitian

    fermionic: bool
        default = False
        use while pllying a SWAP gate. True for fermionic systems.
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    bi_orth: bool
        default=True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors. 
            
    NA: bool
        default=None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.    
        Option for exponentiation = exp(). 
            
    opts_svd: dict
        default=None
        options for truncation on virtual d.o.f.

    optsK_svd: dict
        default=None
        options for truncation on auxilliary d.o.f.

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """
    # change. adjust the sign according to the convention
    sgn = 1j * ( np.sign(dt.real) + 1j * np.sign(dt.imag) )
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
            if not H:
                # canonize and save
                if opts_svd != None:
                    U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), sU=-1, **opts_svd)
                    psi.A[n] = U
                    psi.pC = (n,n1)
                    psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                else:
                    psi.A[n] = init
                    psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)
                
        if H: # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(v, n, conj=True), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2(v, n), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            A1, S, A2 = init[0].split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = A2.dot_diag(S, axis=psi.left)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        if H and n!=psi.g.sweep(to='last', dl=1)[-1]: # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
            init = psi.A[n1]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n1), init=[init], Bv=lambda v : env.Heff1(v, n1, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n1), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n1] = init[0]
            
    for n in psi.g.sweep(to='first', df=1):
        if H and n!=psi.g.sweep(to='first', df=1)[-1]: # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], Bv=lambda v : env.Heff1(v, n, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]

        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
            if not H: # canonize and save
                if opts_svd != None:
                    U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), sU=-1, **opts_svd)
                    psi.A[n] = V
                    psi.pC = (n1,n)
                    psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
                else:
                    psi.A[n] = init
                    psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)
        
        if H: # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2(v, n1), Bv=lambda v: env.Heff2(v, n1, conj=True), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2(v, n1), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            A1, S, A2 = init[0].split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.dot_diag(S, axis=psi.right)
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env



def tdvp_sweep_2site_group(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r""" 
    Perform sweep with 2site-TDVP with grouping neigbouring sites. 
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        initial state.
        can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        operator given in MPO decomposition. 
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    M: Mps, nr_phys=1
        Kraus operators.
        legs are [Kraus dimension, ket-physical, bra-physical]

    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
        
    dt: double
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution 

    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        is MPO hermitian

    fermionic: bool
        default = False
        use while pllying a SWAP gate. True for fermionic systems.
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    bi_orth: bool
        default=True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors. 
            
    NA: bool
        default=None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.    
        Option for exponentiation = exp(). 
            
    opts_svd: dict
        default=None
        options for truncation on virtual d.o.f.

    optsK_svd: dict
        default=None
        options for truncation on auxilliary d.o.f.

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """
    # change. adjust the sign according to the convention
    sgn = 1j * ( np.sign(dt.real) + 1j * np.sign(dt.imag) )
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
            if not H:
                # canonize and save
                if opts_svd != None:
                    U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), sU=-1, **opts_svd)
                    psi.A[n] = U
                    psi.pC = (n,n1)
                    psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                else:
                    psi.A[n] = init
                    psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)
                
        if H: # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
            init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(v, n, conj=True), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2_group(v, n), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            init = init[0].ungroup_leg(axis=1, leg_order=leg_order)
            A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = A2.dot_diag(S, axis=psi.left)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        if H and n!=psi.g.sweep(to='last', dl=1)[-1]: # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
            init = psi.A[n1]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n1), init=[init], Bv=lambda v : env.Heff1(v, n1, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n1), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n1] = init[0]
            
    for n in psi.g.sweep(to='first', df=1):
        if H and n!=psi.g.sweep(to='first', df=1)[-1]: # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], Bv=lambda v : env.Heff1(v, n, conj=True), dt=-dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v : env.Heff1(v, n), init=[init], dt=-dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]

        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK_svd) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
            if not H: # canonize and save
                if opts_svd != None:
                    U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), sU=-1, **opts_svd)
                    psi.A[n] = V
                    psi.pC = (n1,n)
                    psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
                else:
                    psi.A[n] = init
                    psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)
        
        if H: # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
            init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2_group(v, n1), Bv=lambda v: env.Heff2_group(v, n1, conj=True), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2_group(v, n1), init=[init], dt=+dt * .5, tol=eigs_tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            init = init[0].ungroup_leg(axis=1, leg_order=leg_order)
            A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.dot_diag(S, axis=psi.right)
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env
