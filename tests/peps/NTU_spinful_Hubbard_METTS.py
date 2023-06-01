import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_spinful_random, initialize_Neel_spinful, initialize_post_sampling_spinful
from yastn.tn.fpeps.ctm import sample, nn_avg, ctmrg, one_site_avg

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, sym, mu_up, mu_dn, U, t_up, t_dn, beta_target, dbeta, chi, num_iter, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn

    psi = initialize_Neel_spinful(fc_up, fc_dn, fcdag_up, fcdag_dn, net)
    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    gates = gates_homogeneous(psi, g_nn, g_loc)

    mdata = {}
    dbeta = 0.005
    num_steps = int(np.round(beta_target/dbeta))
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_target_beta_%1.3f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[0], dims[1], boundary, purification, fix_bd, beta_target, tr_mode, step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for itr in range(num_iter):

        for num in range(num_steps):

            beta = (num+1)*dbeta
            logging.info("beta = %0.3f" % beta)
            psi, info =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
            print(info)
            for ms in net.sites():
                logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
                xs = psi[ms].unfuse_legs((0, 1))
                for l in range(4):
                    print(xs.get_leg_structure(axis=l))

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
                f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_up, ntu_error_dn))
            with open("SVD_error_ground_state_%s.txt" % file_name, "a+") as f:
                f.write('{:.3f} {:.3e} {:.3e} \n'.format(beta, svd_error_up, svd_error_dn))

        # save the tensor at target beta
        x = {itr: psi[ms].save_to_dict() for ms in psi.sites()}
        mdata.update(x)
        np.save("METTS_fermi_sea_spinless_tensors_target_beta_%1.1f_%s.npy" % (beta_target, file_name), mdata)

        # calculate observables with ctm 

        tol = 1e-10 # truncation of singular values of CTM projectors
        max_sweeps=100  # ctm param
        tol_exp = 1e-5
        opts_svd_ctm = {'D_total': chi, 'tol': tol}

        cf_energy_old = 0

        ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 3rd step as iteration_step=3
            
            obs_hor, obs_ver =  nn_avg(psi, step.env, ops)
            cdagc_up = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
            ccdag_up = 0.5*(abs(obs_hor.get('ccdag_up')) + abs(obs_ver.get('ccdag_up')))
            cdagc_dn = 0.5*(abs(obs_hor.get('cdagc_dn')) + abs(obs_ver.get('cdagc_dn')))
            ccdag_dn = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
            
            mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites


            cf_energy =  U * mean_int - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) 

            print("Energy : ", cf_energy)
            if abs(cf_energy - cf_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            cf_energy_old = cf_energy

        ob_hor, ob_ver = nn_avg(psi, step.env, ops)

        nn_CTM = 0.5 * (abs(ob_hor.get('cdagc')) + abs(ob_ver.get('ccdag')))

        with open("energy_spinless_target_beta_%1.1f_%s.txt" % (beta_target,file_name), "a+") as f:
                f.write('{:.1f} {:.5f}\n'.format(beta, nn_CTM))
    
        # now we do probabilistic sampling
        n_up = fcdag_up @ fc_up 
        n_dn = fcdag_dn @ fc_dn 
        h_up = fc_up @ fcdag_up 
        h_dn = fc_dn @ fcdag_dn 

        nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, n_dn @ h_up, n_up @ n_dn, h_up @ h_dn
        projectors = [nn_up, nn_dn, nn_do, nn_hole]
        out = sample(psi, step.env, projectors)

        psi = initialize_post_sampling_spinful(fc_up, fc_dn, fcdag_up, fcdag_dn, net, out)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=4)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=4)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='True') # purifciation can be 'True' or 'False' or 'Time'
    parser.add_argument("-D", type=int, default=5)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yastn.fpeps.operators.import_distribution
    parser.add_argument("-S", default='U1')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M", type=float, default=0.0)      # chemical potential 
    parser.add_argument("-T", type=float, default=1.0)          # tunelling strength
    parser.add_argument("-BT", type=float, default=0.1)        # target inverse temperature beta
    parser.add_argument("-DBETA", type=float, default=0.005)      # dbeta
    parser.add_argument("-X", type=int, default=20)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-ITER", type=int, default=10)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-STEP", default='two-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    benchmark_NTU_hubbard(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, sym=args.S, dbeta=args.DBETA,
                                mu=args.M, beta_target=args.BT, t=args.T, num_iter=args.ITER, purification=args.p,
                                chi=args.X, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 7-13 nohup python -u NTU_spinless_METTS.py -L 'rectangle' -B 'finite' -p 'True' -x 4 -y 4 -D 5 -X 20 -S 'U1' -M 0 -T 1.0 -DBETA 0.005 -BT 0.2 -ITER 10 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > METTS_spinless_finite_9_9_MU_0_D_5.out &
