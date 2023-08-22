import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as fpeps
import time
from yastn.tn.fpeps.operators.import_distribution import import_distribution
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification
from yastn.tn.fpeps.ctm import nn_avg, ctmrg, one_site_avg, Local_CTM_Env, EV2ptcorr

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def CtmEnv_Hubbard(lattice, boundary, purification, xx, yy, D, sym, chi, step, beta, U, mu_up, mu_dn, t_up, t_dn, tr_mode, fix_bd):

    dims = (xx, yy)
    net = fpeps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    tot_sites = xx * yy
    
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("purification_Hubbard_spinfull_tensors_%s.npy" % (file_name), allow_pickle=True).item()

    sv_beta = round(beta * yastn.BETA_MULTIPLIER)
    psi = fpeps.Peps(lattice, dims, boundary)
    for ms in psi.sites():
        psi[ms] =  yastn.load_from_dict(config=fid.config, d=state.get((ms, sv_beta))) 
    print('BETA: ', beta)

    dict_list_env = []
    for ms in net.sites():
        dict_list_env.extend([('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)])

    state1 = np.load("ctm_environment_beta_%1.0f_%s.npy" % (beta, file_name), allow_pickle=True).item()
    env = {ind: yastn.load_from_dict(config=fid.config, d=state1[ind]) for ind in dict_list_env}
    env1 = {}
    for ms in net.sites():
        env1[ms] = Local_CTM_Env()
        env1[ms].tl = env['cortl',ms] 
        env1[ms].tr = env['cortr',ms] 
        env1[ms].bl = env['corbl',ms] 
        env1[ms].br = env['corbr',ms] 
        env1[ms].t = env['strt',ms] 
        env1[ms].l = env['strl',ms] 
        env1[ms].r = env['strr',ms] 
        env1[ms].b = env['strb',ms]

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}
    
    mean_magnetization, mat_mag = one_site_avg(psi, step.env, 0.5*(n_up-n_dn))
    mean_density, mat_density = one_site_avg(psi, step.env, (n_up+n_dn))
    mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

    obs_hor_mean, obs_ver_mean, obs_hor_sum, obs_ver_sum =  nn_avg(psi, step.env, ops)
    cdagc_up = obs_hor_sum.get('cdagc_up') + obs_ver_sum.get('cdagc_up')
    ccdag_up = - obs_hor_sum.get('ccdag_up') - obs_ver_sum.get('ccdag_up')
    cdagc_dn = obs_hor_sum.get('cdagc_dn') + obs_ver_sum.get('cdagc_dn')
    ccdag_dn = - obs_hor_sum.get('ccdag_dn') - obs_ver_sum.get('ccdag_dn')

    mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites
    tot_energy =  U * mean_int - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn)/tot_sites

    opss = {'SzSz':{'l':0.5*(n_up-n_dn), 'r':0.5*(n_up-n_dn)},
        'nn':{'l':(n_up+n_dn), 'r':(n_up+n_dn)}}
    print("BETA =", beta)
    print("average magnetization ", mean_magnetization)
    print("average density ", mean_density)
    print("Energy : ", tot_energy)
    print("average double occupancy ", mean_int)

    zz_CTM_long_range_1 =  EV2ptcorr(psi, step.env, opss['SzSz'], site0=(0,0), site1=(0,7))
    zz_CTM_long_range_2 =  EV2ptcorr(psi, step.env, opss['SzSz'], site0=(1,0), site1=(1,7))
    zz_CTM_long_range = np.vstack((zz_CTM_long_range_1, zz_CTM_long_range_2))
    print(zz_CTM_long_range)
    np.savetxt("zz_correlation_beta_%1.1f_chi_%1.0f_%s.txt" % (beta, chi, file_name), zz_CTM_long_range)

    site_0 = (0,0)
    site_1 = (0,15)

    site_2 = (1,0)
    site_3 = (1,15)

    print('nn')
    nn_CTM_long_range_1 =  EV2ptcorr(psi, step.env, opss['nn'], site0=site_0, site1=site_1)
    print(nn_CTM_long_range_1)
    nn_CTM_long_range_2 =  EV2ptcorr(psi, step.env, opss['nn'], site0=site_2, site1=site_3)
    print(nn_CTM_long_range_2)
    nn_CTM_long_range_1st_row = np.zeros((16))
    nn_CTM_long_range_2nd_row = np.zeros((16))

    my=2*yy-1
    for ys in range(my):
        nn_CTM_long_range_1st_row[ys] = nn_CTM_long_range_1[ys] - mat_density[0,0]*mat_density[0, ys % 8]
        nn_CTM_long_range_2nd_row[ys] = nn_CTM_long_range_2[ys] - mat_density[1,0]*mat_density[1, ys % 8]

    np.savetxt("nn_correlation_1st_row_beta_%1.1f_chi_%1.0f_%s.txt" % (beta, chi, file_name), nn_CTM_long_range_1st_row)
    np.savetxt("nn_correlation_2nd_row_beta_%1.1f_chi_%1.0f_%s.txt" % (beta, chi, file_name), nn_CTM_long_range_2nd_row)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=2)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=5)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='infinite') # boundary
    parser.add_argument("-p", type=str, default='True') # bool
    parser.add_argument("-D", type=str, default=12) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1xZ2') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=int, default=24) # chi_multiple
    parser.add_argument("-I", type=float, default=0.1) # interval
    parser.add_argument("-BETA_START", type=float, default=0.2) # location
    parser.add_argument("-BETA_END", type=float, default=0.2) # location
    parser.add_argument("-U", type=float, default=0.875)       # location                                                                                             
    parser.add_argument("-MU_UP", type=float, default=-2.2)   # location                                                                                                 
    parser.add_argument("-MU_DOWN", type=float, default=-2.2) # location                                                                                           
    parser.add_argument("-T_UP", type=float, default=1.)    # location
    parser.add_argument("-T_DOWN", type=float, default=1.)  # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    CtmEnv_Hubbard(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, U=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, 
     t_up = args.T_UP, t_dn = args.T_DOWN, ntu_step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)

# to run cprofile : taskset -c 0-6 nohup python -u generate_environment_Hubbard.py -L 'rectangle' -x 6 -y 6 -B 'finite' -p 'True' -D 20 -S 'U1xU1xZ2' -X 100 -I 1 -BETA_START 4 -BETA_END 4 -U 8 -MU_UP 0.0 -MU_DOWN 0.0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > env_D_20_nx_6_ny_6_beta_4.out &
# nohup python -m cProfile -o env_spinful_checkerboard_D_10_U1xU1xZ2.cproof generate_environment_Hubbard_checkerboard.py -L 'checkerboard' -x 2 -y 2 -B 'infinite' -p 'True' -D 10 -S 'U1xU1xZ2' -X 50 -I 1 -BETA_START 4 -BETA_END 4 -U 8 -MU_UP -2.2 -MU_DOWN -2.2 -T_UP 1 -T_DOWN 1 -STEP 'one-step' -MODE 'optimal' -FIXED 1 > env_D_10.out &