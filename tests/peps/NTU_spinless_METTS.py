import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_fermi_sea
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_spinless_random, initialize_post_sampling_spinless
from yastn.tn.fpeps.ctm import sample, nn_avg, ctmrg, one_site_avg

try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, sym, mu, t, beta_target, dbeta, chi, num_iter, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, purification=purification)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, purification=purification) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]

    psi = initialize_spinless_random(fc, fcdag, net) # initialized at infinite temperature
    gates = gates_homogeneous(psi, g_nn, g_loc)

    mdata = {}
    num_steps = int(np.round(beta_target/dbeta))
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[0], dims[1], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for itr in range(num_iter):
        print(itr)

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
            ntu_error = np.mean(np.sqrt(info['ntu_error'][::]))
            logging.info('ntu error : %.2e' % ntu_error)

            svd_error = np.mean(np.sqrt(info['svd_error'][::]))
            logging.info('svd error : %.2e' % svd_error)

            with open("NTU_error_ground_state_target_beta_%1.2f_%s.txt" % (beta_target, file_name), "a+") as f:
                f.write('{:.3f} {:.3e} \n'.format(beta, ntu_error))
            with open("SVD_error_ground_state_target_beta_%1.2f_%s.txt" % (beta_target, file_name), "a+") as f:
                f.write('{:.3f} {:.3e} \n'.format(beta, svd_error))

        # save the tensor at target beta
        x = {itr: psi[ms].save_to_dict() for ms in psi.sites()}
        mdata.update(x)
        np.save("METTS_Hubbard_spinful_tensors_target_beta_%1.2f_%s.npy" % (beta_target, file_name), mdata)

        # calculate observables with ctm 

        tol = 1e-10 # truncation of singular values of CTM projectors
        max_sweeps=100  # ctm param
        tol_exp = 1e-5
        opts_svd_ctm = {'D_total': chi, 'tol': tol}

        cf_energy_old = 0

        ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 3rd step as iteration_step=3
            
            obs_hor, obs_ver =  nn_avg(psi, step.env, ops)
            cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
            ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))

            cf_energy = - (cdagc + ccdag) * 0.5

            print("Energy : ", cf_energy)
            if abs(cf_energy - cf_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            cf_energy_old = cf_energy
        
        with open("energy_from_METTS_spinless_target_beta_%1.2f_%s.txt" % (beta_target,file_name), "a+") as f:
                f.write('{:.0f} {:.5f}\n'.format(itr+1, cf_energy))

        # now we do probabilistic sampling
        nn, hh = fcdag @ fc, fc @ fcdag
        projectors = [nn, hh]
        out = sample(psi, step.env, projectors)
        psi = initialize_post_sampling_spinless(fc, fcdag, net, out)
        


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='True') # purifciation can be 'True' or 'False' or 'Time'; 'True' in case of METTS
    parser.add_argument("-D", type=int, default=4)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yastn.fpeps.operators.import_distribution
    parser.add_argument("-S", default='U1')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M", type=float, default=0.0)      # chemical potential 
    parser.add_argument("-T", type=float, default=1.0)          # tunelling strength
    parser.add_argument("-BT", type=float, default=1.0)        # target inverse temperature beta
    parser.add_argument("-DBETA", type=float, default=0.01)      # dbeta
    parser.add_argument("-X", type=int, default=16)        # chi --- environmental bond dimension for CTM
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


# to run, type in terminal : taskset -c 14-20 nohup python -u NTU_spinless_METTS.py -L 'rectangle' -B 'finite' -p 'True' -x 6 -y 6 -D 7 -X 35 -S 'U1' -M 0 -T 1.0 -DBETA 0.005 -BT 4 -ITER 400 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > METTS_spinless_finite_6_6_MU_0_D_7_BETA_4.out &
