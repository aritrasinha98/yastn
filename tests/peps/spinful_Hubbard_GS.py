import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification, initialize_Neel_spinful
from yastn.tn.fpeps.ctm import nn_avg, ctmrg, one_site_avg

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, sym, mu_up, mu_dn, U, t_up, t_dn, chi, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature
        print('yes')
    elif purification == 'False':
        psi = initialize_Neel_spinful(fc_up, fc_dn, fcdag_up, fcdag_dn, net) # initialized in Néel configuration
        print('no')
    
    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    mdata = {}

    dbeta = 0.1

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_initial_dbeta_%1.3f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, dbeta, tr_mode, step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    beta = 0

    tol = 1e-10 # truncation of singular values of CTM projectors
    max_sweeps=100  # ctm param
    tol_exp = 1e-6 
    energy_old = 0

    for _ in range(20000):

        beta = beta + dbeta
        sv_beta = int(beta * yastn.BETA_MULTIPLIER)
        logging.info("beta = %0.3f" % beta)

        GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
        GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
        g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
        g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

        gates = gates_homogeneous(psi, g_nn, g_loc)
         
        psi, info =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
        print(info)

        for ms in net.sites():
            logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
            xs = psi[ms].unfuse_legs((0, 1))
            for l in range(4):
                print(xs.get_leg_structure(axis=l))
            
        opts_svd_ctm = {'D_total': chi, 'tol': tol}

        ctm_energy_old = 0

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 3rd step as iteration_step=3
            """with open("ctm_time_sweep_beta_%1.1f_chi_%1.1f_%s.txt" % (beta, chi, file_name),"a+") as fs:
                fs.write("{} {}\n".format(step.sweeps, step.tt))"""
              
            mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites
            obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

            kin_hor = abs(obs_hor.get('cdagc_up'))+ abs(obs_hor.get('ccdag_up'))+abs(obs_hor.get('cdagc_dn'))+abs(obs_hor.get('ccdag_dn'))
            kin_ver = abs(obs_ver.get('cdagc_up'))+ abs(obs_ver.get('ccdag_up'))+abs(obs_ver.get('cdagc_dn'))+abs(obs_ver.get('ccdag_dn'))

            ctm_energy = U * mean_int * (xx + yy)  - (kin_hor * (yy-1) * xx  + kin_ver * (xx-1) * yy) 
           
            print("expectation value: ", ctm_energy)
            if abs(ctm_energy - ctm_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            ctm_energy_old = ctm_energy


        print('energy: ', ctm_energy)
        if ctm_energy > energy_old:
            beta = beta_old
            psi = psi_old
            dbeta = dbeta/2.0
            continue

        x = {(ms, sv_beta): psi[ms].save_to_dict() for ms in psi.sites()}
        mdata.update(x)
        np.save("neel_initialized_ground_state_Hubbard_spinful_tensors_%s.npy" % (file_name), mdata)
        if step=='svd-update':
            continue
        ntu_error_up = np.mean(np.sqrt(info['ntu_error'][::2]))
        ntu_error_dn = np.mean(np.sqrt(info['ntu_error'][1::2]))
        logging.info('ntu error up: %.2e' % ntu_error_up)
        logging.info('ntu error dn: %.2e' % ntu_error_dn)

        svd_error_up = np.mean(np.sqrt(info['svd_error'][::2]))
        svd_error_dn = np.mean(np.sqrt(info['svd_error'][1::2]))
        logging.info('svd error up: %.2e' % svd_error_up)
        logging.info('svd error dn: %.2e' % svd_error_dn)

        with open("NTU_error_ground_state_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.2e} {:.2e} {:+.6f}\n'.format(beta, ntu_error_up, ntu_error_dn, ctm_energy))
        with open("SVD_error_ground_state_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.2e} {:.2e} {:+.6f}\n'.format(beta, svd_error_up, svd_error_dn, ctm_energy))

        energy_old = ctm_energy
        psi_old = psi
        beta_old = beta

    
if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=4)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yaps.operators.import_distribution
    parser.add_argument("-S", default='U1xU1_ind')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M_UP", type=float, default=0.0)      # chemical potential up
    parser.add_argument("-M_DOWN", type=float, default=0.0)    # chemical potential down
    parser.add_argument("-U", type=float, default=12.)          # hubbard interaction
    parser.add_argument("-TUP", type=float, default=1.)        # hopping_up
    parser.add_argument("-TDOWN", type=float, default=1.)      # hopping_down
    parser.add_argument("-X", type=int, default=20)   # dbeta
    parser.add_argument("-STEP", default='two-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    benchmark_NTU_hubbard(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, sym=args.S, 
                                mu_up=args.M_UP, mu_dn=args.M_DOWN, U=args.U, t_up=args.TUP, t_dn=args.TDOWN, purification=args.p,
                                chi=args.X, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 7-13 nohup python -u spinful_Hubbard_ground_GS.py -L 'rectangle' -B 'finite' -p 'False' -x 5 -y 5 -D 6 -X 30 -S 'U1xU1xZ2' -M_UP 0 -M_DOWN 0 -U 12 -TUP 1 -TDOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > ground_state_neel_initialized_spinfull_finite_9_9_MU_0_U_12_D_9.out &
