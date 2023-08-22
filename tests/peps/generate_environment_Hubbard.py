# this script analyzes the saved peps tensor, calculates expectation values using CTM and error analysis
import numpy as np
import logging
import argparse
import time
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.ctm import nn_avg, ctmrg, one_site_avg, EV2ptcorr

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def CtmEnv_Hubbard(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_end, beta_start, U, mu_up, mu_dn, t_up, t_dn, ntu_step, tr_mode, fix_bd):

    dims = (xx, yy)
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    tot_sites = xx * yy

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, ntu_step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("purification_Hubbard_spinful_tensors_%s.npy" % (file_name), allow_pickle=True).item()

    num_ints = round((beta_end-beta_start)/interval)+1
    beta_range = np.linspace(beta_start, beta_end, num_ints)
    print(beta_range)
    
    for beta in beta_range:
        
        sv_beta = round(beta * yastn.BETA_MULTIPLIER)
        psi = fpeps.Peps(lattice, dims, boundary)
        for ms in psi.sites():
            psi[ms] =  yastn.load_from_dict(config=fid.config, d=state.get((ms, sv_beta))) 
        print('BETA: ', beta)
     
        tol = 1e-10 # truncation of singular values of CTM projectors
        max_sweeps=100 
        tol_exp = 1e-6   # difference of some observable must be lower than tolernace

        ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

        tot_energy_old = 0
        opts_svd_ctm = {'D_total': chi, 'tol': tol}

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 2nd step as iteration_step=2
            """with open("ctm_time_sweep_beta_%1.1f_chi_%1.1f_%s.txt" % (beta, chi, file_name),"a+") as fs:
                fs.write("{} {}\n".format(step.sweeps, step.tt))"""
              
            mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

            obs_hor_mean, obs_ver_mean, obs_hor_sum, obs_ver_sum =  nn_avg(psi, step.env, ops)
            cdagc_up = obs_hor_sum.get('cdagc_up') + obs_ver_sum.get('cdagc_up')
            ccdag_up = - obs_hor_sum.get('ccdag_up') - obs_ver_sum.get('ccdag_up')
            cdagc_dn = obs_hor_sum.get('cdagc_dn') + obs_ver_sum.get('cdagc_dn')
            ccdag_dn = - obs_hor_sum.get('ccdag_dn') - obs_ver_sum.get('ccdag_dn')

            mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites
            tot_energy =  U * mean_int - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn)/tot_sites 

            if abs(tot_energy - tot_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            tot_energy_old = tot_energy

        mdata={}                                                                                 
        for ms in psi.sites():
            xm = {('cortl', ms, sv_beta): step.env[ms].tl.save_to_dict(), ('cortr', ms, sv_beta): step.env[ms].tr.save_to_dict(),
            ('corbl', ms, sv_beta): step.env[ms].bl.save_to_dict(), ('corbr', ms, sv_beta): step.env[ms].br.save_to_dict(),
            ('strt', ms, sv_beta): step.env[ms].t.save_to_dict(), ('strb', ms, sv_beta): step.env[ms].b.save_to_dict(),
            ('strl', ms, sv_beta): step.env[ms].l.save_to_dict(), ('strr', ms, sv_beta): step.env[ms].r.save_to_dict()}
            mdata.update(xm)
  
        with open("ctm_environment_beta_%1.1f_chi_%1.1f_%s.npy" % (beta, chi, file_name), 'wb') as f:
            np.save(f, mdata, allow_pickle=True)

        mean_magnetization, mat_mag = one_site_avg(psi, step.env, 0.5*(n_up-n_dn))
        mean_density, mat_density = one_site_avg(psi, step.env, (n_up+n_dn))

        print("BETA =", beta)
        print("average magnetization ", mean_magnetization)
        print("average density ", mean_density)
        print("Energy : ", tot_energy)
        print("average double occupancy ", mean_int)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='checkerboard') # lattice shape
    parser.add_argument("-x", type=int, default=2)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=2)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='infinite') # boundary
    parser.add_argument("-p", type=str, default='True') # bool
    parser.add_argument("-D", type=str, default=12) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1xZ2') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=int, default=24) # chi_multiple
    parser.add_argument("-I", type=float, default=0.1) # interval
    parser.add_argument("-BETA_START", type=float, default=1) # location
    parser.add_argument("-BETA_END", type=float, default=1) # location
    parser.add_argument("-U", type=float, default=8)       # location                                                                                             
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

# to run cprofile : taskset -c 0-6 nohup python -u generate_environment_Hubbard.py -L 'checkerboard' -x 2 -y 2 -B 'infinite' -p 'True' -D 10 -S 'U1xU1xZ2' -X 100 -I 1 -BETA_START 4 -BETA_END 4 -U 8 -MU_UP 0.0 -MU_DOWN 0.0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > env_D_20_nx_6_ny_6_beta_4.out &
# nohup python -m cProfile -o env_spinful_checkerboard_D_10_U1xU1xZ2.cproof generate_environment_Hubbard_checkerboard.py -L 'checkerboard' -x 2 -y 2 -B 'infinite' -p 'True' -D 10 -S 'U1xU1xZ2' -X 50 -I 1 -BETA_START 4 -BETA_END 4 -U 8 -MU_UP -2.2 -MU_DOWN -2.2 -T_UP 1 -T_DOWN 1 -STEP 'one-step' -MODE 'optimal' -FIXED 1 > env_D_10.out &