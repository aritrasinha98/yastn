"""
Routines for CTMRG of layers PEPS on an arbitrary mxn rectangular lattice.

PEPS tensors A[0] and A[1] corresponds to "black" and "white" sites of the checkerboard.
PEPs tensors habe 6 legs: [top, left, bottom, right, ancilla, system] with signature s=(-1, 1, 1, -1, -1, 1).
The routines include swap_gates to incorporate fermionic anticommutation relations into the lattice.
"""

import yast
from yast import tensordot, ncon, svd_with_truncation, qr, vdot
from yast.tn.peps._doublePepsTensor import DoublePepsTensor
import numpy as np
from .CtmEnv import CtmEnv
try:
    from .configs import config_dense 
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense

def append_a_bl(tt, AAb):
    """
    ten indices counterclockwise order.
    A indices counterclockwise order starting from top, then spin and ancilla indices.
    tt indices counterclockwise order (e1,[t1,b1],e2,[t2,b2]).
    """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_bl(tt)
    return append_a_bl2(tt, AAb)


def append_a_bl2(tt, AAb):
    # tt: e3 (2t 2b) (1t 1b) e0
    # AAb['bl']: ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))
    tt = tt.fuse_legs(axes=((0, 3), (1, 2)))  # (e3 e0) ((2t 2b) (1t 1b))
    tt = tensordot(tt, AAb['bl'], axes=(1, 0))
    tt = tt.unfuse_legs(axes=(0, 1))  # e3 e0 (0t 0b) (3t 3b)
    tt = tt.transpose(axes=(0, 3, 1, 2)) # e3 (3t 3b) e0 (0t 0b)
    return tt

def append_a_tr(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_tr(tt)
    return append_a_tr2(tt, AAb)


def append_a_tr2(tt, AAb):
    # tt:  e1 (0t 0b) (3t 3b) e2
    # AAb['bl']: ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))
    tt = tt.fuse_legs(axes=((1, 2), (0, 3)))  # ((0t 0b) (3t 3b)) (e1 e2)
    tt = tensordot(AAb['bl'], tt, axes=(1, 0))  # ((2t 2b) (1t 1b)) (e1 e2)
    tt = tt.unfuse_legs(axes=(0, 1))  # (2t 2b) (1t 1b) e1 e2
    tt = tt.transpose(axes=(2, 1, 3, 0))  # e1 (1t 1b) e2 (2t 2b)
    return tt

def append_a_tl(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_tl(tt)
    return append_a_tl2(tt, AAb)


def append_a_tl2(tt, AAb):
    # tt: e2 (1t 1b) (0t 0b) e3
    # AAb['tl']:  ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
    tt = tt.fuse_legs(axes=((0, 3), (1, 2)))  # (e2 e3) ((1t 1b) (0t 0b))
    tt = tensordot(tt, AAb['tl'], axes=(1, 0))  
    tt = tt.unfuse_legs(axes=(0, 1))  # e2 e3 (3t 3b) (2t 2b)
    tt = tt.transpose(axes=(0, 3, 1, 2))  # (e2 (2t 2b)) (e3 (3t 3b))
    return tt


def append_a_br(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_br(tt)
    return append_a_br2(tt, AAb)

def append_a_br2(tt, AAb):
    # tt: e0 (3t 3b) (2t 2b) e1
    # AAb['tl']:  ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
    tt = tt.fuse_legs(axes=((1, 2), (0, 3)))  # ((3t 3b) (2t 2b)) (e0 e1)
    tt = tensordot(AAb['tl'], tt, axes=(1, 0))  # ((1t 1b) (0t 0b)) (e0 e1)
    tt = tt.unfuse_legs(axes=(0, 1))  # (1t 1b) (0t 0b) e0 e1
    tt = tt.transpose(axes=(2, 1, 3, 0))  # e0 (0t 0b) e1 (1t 1b)
    return tt

def fcor_bl(env, AAb, ind):
    """ Creates extended bottom left corner. Order of indices see append_a_bl. """
    corbln = tensordot(env[ind].b, env[ind].bl, axes=(2, 0))
    corbln = tensordot(corbln, env[ind].l, axes=(2, 0))
    return append_a_bl(corbln, AAb)

def fcor_tl(env, AAb, ind):
    """ Creates extended top left corner. """
    cortln = tensordot(env[ind].l, env[ind].tl, axes=(2, 0))
    cortln = tensordot(cortln, env[ind].t, axes=(2, 0))
    return append_a_tl(cortln, AAb)

def fcor_tr(env, AAb, ind):
    """ Creates extended top right corner. """
    cortrn = tensordot(env[ind].t, env[ind].tr, axes=(2, 0))
    cortrn = tensordot(cortrn, env[ind].r, axes=(2, 0))
    return append_a_tr(cortrn, AAb)

def fcor_br(env, AAb, ind):
    """ Creates extended bottom right corner. """
    corbrn = tensordot(env[ind].r, env[ind].br, axes=(2, 0))
    corbrn = tensordot(corbrn, env[ind].b, axes=(2, 0))
    return append_a_br(corbrn, AAb)


def apply_TMO_left(vecl, env, indten, indop, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[indten].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb[indop])
    new_vecl = ncon((new_vecl, env[indten].b), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecl = new_vecl.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecl.ndim == 5:
        new_vecl = new_vecl.swap_gate(axes=((0, 3), 2))
        new_vecl = new_vecl.fuse_legs(axes=((0, 2), (1, 3), 4))
    elif new_vecl.ndim == 4:
        new_vecl =  new_vecl.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecl


def apply_TMO_right(vecr, env, indten, indop, AAb):
    """ apply TM (top-AAB-bottom) to right boundary vector"""
    new_vecr = tensordot(vecr, env[indten].b, axes=(2, 0))
    new_vecr = append_a_br(new_vecr, AAb[indop])
    new_vecr = ncon((new_vecr, env[indten].t), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecr = new_vecr.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecr.ndim == 5:
        new_vecr = new_vecr.swap_gate(axes=((3, 4), 2))
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vecr.ndim == 4:
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecr


def apply_TM_left(vecl, env, indten, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[indten].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb)
    new_vecl = new_vecl.unfuse_legs(axes=0)
    if new_vecl.ndim == 5:
        new_vecl = ncon((new_vecl, env[indten].b), ([1, -4, 2, -3, -2], [-1, 2, 1]))
        new_vecl = new_vecl.fuse_legs(axes=((0, 3), 1, 2))
    elif new_vecl.ndim == 4:
        new_vecl = ncon((new_vecl, env[indten].b), ([1, 2, -3, -2], [-1, 2, 1]))
    return new_vecl


def apply_TMO_top(vect, env, indten, indop, AAb):
    """ apply TMO (top-AAB-bottom) to top boundary vector"""
    new_vect = tensordot(env[indten].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb[indop])
    new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, 1], [2, 1, -3]))
    new_vect = new_vect.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vect.ndim == 5:
        new_vect = new_vect.swap_gate(axes=((1, 4), 2))
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vect.ndim == 4:
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 2), 3))
    return new_vect


def apply_TMO_bottom(vecb, env, indten, indop, AAb):
    """ apply TMO (bottom-AAB-top) to bottom boundary vector"""
    new_vecb = tensordot(env[indten].r, vecb, axes=(2, 0))
    new_vecb = append_a_br(new_vecb, AAb[indop])
    new_vecb = ncon((new_vecb, env['strl', indten]), ([-1, -2, 1, 2], [1, 2, -3]))
    new_vecb = new_vecb.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecb.ndim == 5:
        new_vecb = new_vecb.swap_gate(axes=((0, 1), 2))
        new_vecb =  new_vecb.fuse_legs(axes=((0, 2), (1, 3), 4)) 
    elif new_vecb.ndim == 4:
        new_vecb =  new_vecb.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecb


def apply_TM_top(vect, env, indten, AAb):
    """ apply TM (left-AAB-right)   to top boundary vector"""
    new_vect = tensordot(env[indten].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb)
    new_vect = new_vect.unfuse_legs(axes=2)
    if new_vect.ndim == 5:
        new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, -4, 1], [2, 1, -3]))
        new_vect = new_vect.fuse_legs(axes=(0, 1, (2, 3)))
    elif new_vect.ndim == 4:
        new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, 1], [2, 1, -3]))
    return new_vect



def EV2sNN_ver(env, indten_t, indten_b, indop_t, indop_b, AAb):
    """
    Calculates 2-site verical NN expectation value.
    indt is the top PEPS index. indb is the bottom PEPS index.
    """
    cortln = fcor_tl(env, indten_t, indop_t, AAb)
    corttn = ncon((cortln, env[indten_t].tr, env[indten_t].r), ((-0, -1, 2, 3), (2, 1), (1, 3, -2)))
    del cortln
    corbrn = fcor_br(env, indten_b, indop_b, AAb)
    corbbn = ncon((corbrn, env[indten_b].l, env[indten_b].bl), ((-2, -1, 2, 3), (1, 3, -0), (2, 1)))
    del corbrn
    return vdot(corttn, corbbn, conj=(0, 0))


def EV1s(env, indop, AAb):
    """
    Calculates 1-site expectation value.
    indt is the top PEPS index. indb is the bottom PEPS index.
    """
    indten = 0 if indop == 'l' else 1
    cortln = fcor_tl(env, indten, indop, AAb)
    top_part = ncon((cortln, env[indten].tr, env[indten].r), ((-0, -1, 2, 3), (2, 1), (1, 3, -2)))
    bot_part = ncon((env[indten].br, env[indten].b, env[indten].bl), ((-2, 1), (1, -1, 2), (2, -0)))
    return vdot(top_part, bot_part, conj=(0, 0))


def proj_Cor(tt, tb, chi, cutoff):
    """
    tt upper half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    tb lower half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    """
    _, rt = qr(tt, axes=((0, 1), (2, 3)))
    _, rb = qr(tb, axes=((0, 1), (2, 3)))

    rr = tensordot(rt, rb, axes=((1, 2), (1, 2)))
    u, s, v = svd_with_truncation(rr, axes=(0, 1), tol=cutoff, D_total=chi)
    s = s.rsqrt()
    pt = s.broadcast(tensordot(rb, v, axes=(0, 1), conj=(0, 1)), axis=2)
    pb = s.broadcast(tensordot(rt, u, axes=(0, 0), conj=(0, 1)), axis=2)
    return pt, pb


def proj_horizontal(env, AAb, chi, cutoff, ms):

    # ms in the CTM unit cell 
    out = {}
    
    cortlm = fcor_tl(env, AAb[ms.nw], ms.nw) # contracted matrix at top-left constructing a projector with cut in the middle
    cortrm = fcor_tr(env, AAb[ms.ne], ms.ne) # contracted matrix at top-right constructing a projector with cut in the middle
    corttm = tensordot(cortrm, cortlm, axes=((0, 1), (2, 3))) # top half for constructing middle projector
    del cortlm
    del cortrm
    
    corblm = fcor_bl(env, AAb[ms.sw], ms.sw) # contracted matrix at bottom-left constructing a projector with cut in the middle
    corbrm = fcor_br(env, AAb[ms.se], ms.se) # contracted matrix at bottom-right constructing a projector with cut in the middle
    corbbm = tensordot(corbrm, corblm, axes=((2, 3), (0, 1))) 
    del corblm
    del corbrm

    out['ph_l_t', ms.nw], out['ph_l_b', ms.nw] = proj_Cor(corttm, corbbm, chi, cutoff)   # projector left-middle
    corttm = corttm.transpose(axes=(2, 3, 0, 1))
    corbbm = corbbm.transpose(axes=(2, 3, 0, 1))
    out['ph_r_t', ms.ne], out['ph_r_b', ms.ne] = proj_Cor(corttm, corbbm, chi, cutoff)   # projector right-middle
    del corttm
    del corbbm

    return out


def proj_vertical(env, AAb, chi, cutoff, ms):

    out = {}
    
    cortlm = fcor_tl(env, AAb[ms.nw], ms.nw) # contracted matrix at top-left constructing a projector with a cut in the middle
    corblm = fcor_bl(env, AAb[ms.sw], ms.sw) # contracted matrix at bottom-left constructing a projector with a cut in the middle

    corvvm = tensordot(corblm, cortlm, axes=((2, 3), (0, 1))) # left half for constructing middle projector
    del corblm
    del cortlm

    cortrm = fcor_tr(env, AAb[ms.ne], ms.ne) # contracted matrix at top-right constructing a projector with cut in the middle
    corbrm = fcor_br(env, AAb[ms.se], ms.se) # contracted matrix at bottom-right constructing a projector with cut in the middle

    corkkr = tensordot(corbrm, cortrm, axes=((0, 1), (2, 3))) # right half for constructing middle projector
    out['pv_t_l', ms.nw], out['pv_t_r', ms.nw] = proj_Cor(corvvm, corkkr, chi, cutoff) # projector top-middle 
    corvvm = corvvm.transpose(axes=(2, 3, 0, 1))
    corkkr = corkkr.transpose(axes=(2, 3, 0, 1))
    out['pv_b_l', ms.sw], out['pv_b_r', ms.sw] = proj_Cor(corvvm, corkkr, chi, cutoff) # projector bottom-middle
    del corvvm
    del corkkr

    return out



def move_horizontal(env, lattice, AAb, proj, ms):
    """ Perform horizontal CTMRG move on a mxn lattice. """
    envn = env.copy()

    nw_abv = lattice.nn_site(ms.nw, d='t')
    ne_abv = lattice.nn_site(ms.ne, d='t')

    envn[ms.ne].tl = ncon((env[ms.nw].tl, env[ms.nw].t, proj['ph_l_t', nw_abv]),
                                   ([2, 3], [3, 1, -1], [2, 1, -0]))

    tt_l = tensordot(env[ms.nw].l, proj['ph_l_b', nw_abv], axes=(2, 0))
    tt_l = append_a_tl(tt_l, AAb[ms.nw])
    envn[ms.ne].l = ncon((proj['ph_l_t', ms.nw], tt_l), ([2, 1, -0], [2, 1, -2, -1]))

    bb_l = ncon((proj['ph_l_t', ms.sw], env[ms.sw].l), ([1, -1, -0], [1, -2, -3]))
    bb_l = append_a_bl(bb_l, AAb[ms.sw])

    envn[ms.se].l = ncon((proj['ph_l_b', ms.nw], bb_l), ([2, 1, -2], [-0, -1, 2, 1]))

    envn[ms.se].bl = ncon((env[ms.sw].bl, env[ms.sw].b, proj['ph_l_b', ms.sw]),
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))

    envn[ms.nw].tr = ncon((env[ms.ne].tr, env[ms.ne].t, proj['ph_r_t', ne_abv]),
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))

    tt_r = ncon((env[ms.ne].r, proj['ph_r_b', ne_abv]), ([1, -2, -3], [1, -1, -0]))
    tt_r = append_a_tr(tt_r, AAb[ms.ne])

    bb_r = tensordot(env[ms.se].r, proj['ph_r_t', ms.se], axes=(2, 0))
    bb_r = append_a_br(bb_r, AAb[ms.se])

    envn[ms.sw].r = tensordot(proj['ph_r_b', ms.ne], bb_r, axes=((1, 0), (1, 0))).fuse_legs(axes=(0, 2, 1))

    
    envn[ms.sw].br = ncon((proj['ph_r_b', ms.se], env[ms.se].br, env[ms.se].b), 
                               ([2, 1, -0], [2, 3], [3, 1, -1]))

    envn[ms.ne].tl = envn[ms.ne].tl / envn[ms.ne].tl.norm(p='inf')
    envn[ms.ne].l = envn[ms.ne].l / envn[ms.ne].l.norm(p='inf')
    envn[ms.se].l = envn[ms.se].l / envn[ms.se].l.norm(p='inf')
    envn[ms.se].bl = envn[ms.se].bl / envn[ms.se].bl.norm(p='inf')
    envn[ms.nw].tr = envn[ms.nw].tr / envn[ms.nw].tr.norm(p='inf')
    envn[ms.nw].r = envn[ms.nw].r / envn[ms.nw].r.norm(p='inf')
    envn[ms.sw].r = envn[ms.sw].r / envn[ms.sw].r.norm(p='inf')
    envn[ms.sw].br = envn[ms.sw].br / envn[ms.sw].br.norm(p='inf')

    return envn



def move_vertical(env, lattice, AAb, proj, ms):
    """ Perform vertical CTMRG on a mxn lattice """

    envn = env.copy()

    nw_left = lattice.nn_site(ms.nw, d='l')
    sw_left = lattice.nn_site(ms.sw, d='l')

    envn[ms.sw].tl = ncon((env[ms.nw].tl, env[ms.nw].l, proj['pv_t_l', nw_left]), 
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))
    
    ll_t = ncon((proj['pv_t_r', nw_left], env[ms.nw].t), ([1, -1, -0], [1, -2, -3]))
    ll_t = append_a_tl(ll_t, AAb[ms.nw])
    envn[ms.sw].t = ncon((ll_t, proj['pv_t_l', ms.nw]), ([-0, -1, 2, 1], [2, 1, -2]))

    rr_t = tensordot(env[ms.ne].t, proj['pv_t_l', ms.ne], axes=(2, 0))
    rr_t = append_a_tr(rr_t, AAb[ms.ne])
    envn[ms.se].t = ncon((proj['pv_t_r', ms.nw], rr_t), ([2, 1, -0], [2, 1, -2, -1]))
    envn[ms.se].tr = ncon((proj['pv_t_r', ms.ne], envn[ms.ne].tr, envn[ms.ne].r), 
                               ([2, 1, -0], [2, 3], [3, 1, -1]))
    
    envn[ms.nw].bl = ncon((env[ms.sw].bl, env[ms.sw].l, proj['pv_b_l', sw_left]), 
                               ([1, 3], [3, 2, -1], [1, 2, -0]))

    ll_b = tensordot(env[ms.sw].b, proj['pv_b_r', sw_left], axes=(2, 0))
    ll_b = append_a_bl(ll_b, AAb[ms.sw])
    envn[ms.nw].b = ncon((ll_b, proj['pv_b_l', ms.sw]), ([2, 1, -2, -1], [2, 1, -0]))

    rr_b = ncon((proj['pv_b_l', ms.se], env[ms.se].b), ([1, -1, -0], [1, -2, -3]))
    rr_b = append_a_br(rr_b, AAb[ms.se])
    envn[ms.ne].b = tensordot(rr_b, proj['pv_b_r', ms.sw], axes=((3, 2), (1, 0)))

    envn[ms.ne].br = ncon((proj['pv_b_r', ms.se], env[ms.se].br, env[ms.se].r), 
                               ([2, 1, -1], [3, 2], [-0, 1, 3]))

    envn[ms.sw].tl = envn[ms.sw].tl/ envn[ms.sw].tl.norm(p='inf')
    envn[ms.sw].t = envn[ms.sw].t/ envn[ms.sw].t.norm(p='inf')
    envn[ms.se].t = envn[ms.se].t/ envn[ms.se].t.norm(p='inf')
    envn[ms.se].tr = envn[ms.se].tr/ envn[ms.se].tr.norm(p='inf')
    envn[ms.nw].bl = envn[ms.nw].bl/ envn[ms.nw].bl.norm(p='inf')
    envn[ms.nw].b = envn[ms.nw].b/ envn[ms.nw].b.norm(p='inf')
    envn[ms.ne].b = envn[ms.ne].b/ envn[ms.ne].b.norm(p='inf')
    envn[ms.ne].br = envn[ms.ne].br/ envn[ms.ne].br.norm(p='inf')

    return envn

def CTM_it(lattice, env, AAb, chi, cutoff):
    r""" 
    Perform one step of CTMRG update for a mxn lattice 

    Procedure
    ---------

    4 ways to move though the lattice using CTM: 'left', 'right', 'top' and 'bottom'.
    Done iteratively on a mxn lattice with 2x2 unit PEPS representations. 
    Example: A left trajectory would comprise a standard CTM move of absorbtion 
    and renormalization starting with a 2x2 cell in the top-left corner and 
    subsequently going downward for (Ny-1) steps and then (Nx-1) steps to the right.      
    """
    proj={}
    proj_hor = {}
    proj_ver = {}

    Ny, Nx = lattice.Ny, lattice.Nx
    CmEv = CtmEnv(lattice)

    for y in range(Ny):
        for ms in CmEv.tensors_CtmEnv(trajectory='h')[y*Nx:(y+1)*Nx]:   # horizontal absorption and renormalization
            print('ctm cluster horizontal', ms)
            proj = proj_horizontal(env, AAb, chi, cutoff, ms)
            proj_hor.update(proj)
        for ms in CmEv.tensors_CtmEnv(trajectory='h')[y*Nx:(y+1)*Nx]:   # horizontal absorption and renormalization
            env = move_horizontal(env, lattice, AAb, proj_hor, ms)

    proj={}
    for x in range(Nx):
        for ms in CmEv.tensors_CtmEnv(trajectory='v')[x*Ny:(x+1)*Ny]:   # vertical absorption and renormalization
            print('ctm cluster vertical', ms)
            proj = proj_vertical(env, AAb, chi, cutoff, ms)
            proj_ver.update(proj)
        for ms in CmEv.tensors_CtmEnv(trajectory='v')[x*Ny:(x+1)*Ny]:   # vertical absorption and renormalization
            env = move_vertical(env, lattice, AAb, proj_ver, ms)       

    return env, proj_hor, proj_ver       


def fPEPS_l(A, op):
    """
    attaches operator to the tensor located at the left (left according to 
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """

    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.swap_gate(axes=(2, 5))
    Aop = Aop.fuse_legs(axes=(0, 1, 2, (3, 5), 4))
    return Aop


def fPEPS_r(A, op):
    """
    attaches operator to the tensor located at the right (right according to 
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """

    """if A.ndim == 6:
        Aop = tensordot(A, op, axes=(5, 0))
        Aop = Aop.fuse_legs(axes=(0, (1, 6), 2, 3, 4, 5))
    elif A.ndim == 5:"""
   # Aop = tensordot(A, op, axes=(4, 0))
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, (1, 5), 2, 3, 4))
    return Aop


def fPEPS_t(A, op):
    """
    attaches operator to the tensor located at the top (left according to 
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    """if A.ndim == 6:
        Aop = tensordot(A, op, axes=(5, 0))
        Aop = Aop.swap_gate(axes=(4, 6))
        Aop = Aop.fuse_legs(axes=(0, 1, (2, 6), 3, 4, 5))
    elif A.ndim == 5:"""
    #Aop = tensordot(A, op, axes=(4, 0))
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, 1, (2, 5), 3, 4))
    return Aop


def fPEPS_b(A, op):
    """
    attaches operator to the tensor located at the bottom (right according to 
    chosen fermionic order) while calulating expectation values of non-local 
    fermionic operators in vertical direction
    """

    """if A.ndim == 6:
        Aop = tensordot(A, op, axes=(5, 0))
        Aop = Aop.swap_gate(axes=(1, 6))
        Aop = Aop.fuse_legs(axes=((0, 6), 1, 2, 3, 4, 5))
    elif A.ndim == 5:"""
    #Aop = tensordot(A, op, axes=(4, 0))
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.swap_gate(axes=(1, 5))
    Aop = Aop.fuse_legs(axes=((0, 5), 1, 2, 3, 4))
    return Aop

def fPEPS_op1s(A, op):
    """
    attaches operator to the tensor while calulating expectation
    values of local operators, no need to fuse auxiliary legs
    """

    """if A.ndim == 6:
        Aop = tensordot(A, op, axes=(5, 0))
    elif A.ndim == 5:"""
   # Aop = tensordot(A, op, axes=(4, 0))
    Aop = tensordot(A, op, axes=(4, 1))
    return Aop

def fPEPS_fuse_physical(A):
    return {ind: x.fuse_legs(axes=(0, 1, 2, 3, (4, 5))) for ind, x in A.items()}

def fPEPS_unfuse_physical(A):
    return {ind: x.unfuse_legs(axes=4) for ind, x in A.items()}

def fuse_ancilla_wos(op, fid):
    """ kron and fusion of local operator with identity for ancilla --- without string """
    op = ncon((op, fid), ((-0, -1), (-2, -3)))
    return op.fuse_legs(axes=((0, 3), (1, 2)))

def fuse_ancilla_ws(op, fid, dirn):
    """ kron and fusion of nn operator with identity for ancilla --- with string """
    if dirn == 'l':
        op= op.add_leg(s=1).swap_gate(axes=(0, 2))
        op = ncon((op, fid), ((-0, -1, -4), (-2, -3)))
        op = op.swap_gate(axes=(2, 4)) # swap of connecting axis with ancilla is always in GA gate
        op = op.fuse_legs(axes=((0, 3), (1, 2), 4))
    elif dirn == 'r':
        op = op.add_leg(s=-1)
        op = ncon((op, fid), ((-0, -1, -4), (-2, -3)))
        op = op.fuse_legs(axes=((0, 3), (1, 2), 4))
    return op



def fPEPS_only_spin(A, B=None, op=None):

    if op is not None:
        B = A.swap_gate(axes=(0, 1, 2, 3))
        A = A.unfuse_legs(axes=4)
        Ao = tensordot(A, op, axes=(4, 1))
        Ao = Ao.fuse_legs(axes=(0, 1, 2, 3, (5, 4)))
        AAb = DoublePepsTensor(top=Ao, btm=B)
    elif op is None:
        Ao = A
        B = A.swap_gate(axes=(0, 1, 2, 3))
        AAb = DoublePepsTensor(top=Ao, btm=B)

    return AAb


def fPEPS_2layers(A, B=None, op=None, dir=None):
    """ 
    Prepare top and bottom peps tensors for CTM procedures.
    Applies operators on top if provided, with dir = 'l', 'r', 't', 'b', '1s'
    If dir = '1s', no auxiliary indices are introduced as the operator is local.
    Here spin and ancilla legs of tensors are fused
    """
    leg = A.get_legs(axis=-1)
    leg, _ = yast.leg_undo_product(leg) # last leg of A is fused 
    fid = yast.eye(config=A.config, legs=[leg, leg.conj()]).diag() # identity generated to act on ancilla leg through the operators

    if op is not None:
        if dir == 't':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_t(A,op_aux)
        elif dir == 'b':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_b(A,op_aux)
        elif dir == 'l':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_l(A,op_aux)
        elif dir == 'r':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_r(A,op_aux)
        elif dir == '1s':
            op = fuse_ancilla_wos(op,fid)
            Ao = fPEPS_op1s(A,op)
        else:
            raise RuntimeError("dir should be equal to 'l', 'r', 't', 'b' or '1s'")        
    else:
        Ao = A
    if B is None:
        B = A.swap_gate(axes=(0, 1, 2, 3))
    elif B is not None:
        B = B.swap_gate(axes=(0, 1, 2, 3))
    AAb = DoublePepsTensor(top=Ao, btm=B)
    return AAb


def fPEPS_fuse_layers(AAb, EVonly=False):
    for st in AAb.values():
        fA = st.top.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0t 1t) (2t 3t) 4t
       # st.pop('A')
        fAb = st.btm.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0b 1b) (2b 3b) 4b
       # st.pop('Ab')
        tt = tensordot(fA, fAb, axes=(2, 2), conj=(0, 1))  # (0t 1t) (2t 3t) (0b 1b) (2b 3b)
        tt = tt.fuse_legs(axes=(0, 2, (1, 3)))  # (0t 1t) (0b 1b) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=(0, 1))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.swap_gate(axes=(1, 2))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((0, 2), (1, 3), 4))  # (0t 0b) (1t 1b) ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((1, 0), 2))  # ((1t 1b) (0t 0b)) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=1)  # ((1t 1b) (0t 0b)) (2t 3t) (2b 3b)
        tt = tt.unfuse_legs(axes=(1, 2))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.swap_gate(axes=(1, 4))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))  # ((1t 1b) (0t 0b)) (2t 2b) (3t 3b)
        st['tl'] = tt.fuse_legs(axes=(0, (2, 1))) # ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
        if not EVonly:
            tt = tt.unfuse_legs(axes=0)  # (1t 1b) (0t 0b) (2t 2b) (3t 3b)
            st['bl'] = tt.fuse_legs(axes=((2, 0), (1, 3)))  # ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))


def check_consistency_tensors(A, net):
    # to check if the A tensors have the appropriate configuration of legs i.e. t l b r [s a]

    list_sites = net.sites()
    if A._data[0, 0].ndim == 6:
        A._data = {ms: A._data[ms].fuse_legs(axes=(0, 1, 2, 3, (4, 5))) for ms in list_sites}  # make sure A is a dict
    elif A._data[0, 0].ndim == 3:
        A._data = {ms: A._data[ms].unfuse_legs(axes=(0, 1)) for ms in list_sites}  # system and ancila are fused by default
    else:
        A._data = {ms: A._data[ms] for ms in list_sites}  # system and ancila are fused by default
    return A
        