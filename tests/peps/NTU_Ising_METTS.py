import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gate_local_dense, gate_Ising
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_dense_random, initialize_post_sampling_Ising
from yastn.tn.fpeps.ctm import sample, nn_avg, ctmrg, one_site_avg, EV2ptcorr

try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_dense as cfg

def NTU_Ising_METTS(lattice, boundary, purification, xx, yy, D, sym, J, hx, hz, beta_target, dbeta, chi, num_iter, step, tr_mode):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns) # gives the lattice info # checkerboard, (2,2), infinite
    opt = yastn.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    sid, sz, sx = opt.I(), opt.z(), opt.x()
    sup = 0.5*(sid+sz)
    sdn = 0.5*(sid-sz)

    psi = initialize_dense_random(sid, sz, net)

    g_loc = gate_local_dense(sid, sx, sz, hx, hz, dbeta, purification=purification)   # \beta_p     
    GA_nn, GB_nn = gate_Ising(sid, sz, J, dbeta, purification=purification)  # gate for 2D fermi sea  # z in the operator sigma_z
    g_nn = [(GA_nn, GB_nn)]        
    gates = gates_homogeneous(psi, g_nn, g_loc) # takes into account both local and NN gates

    mdata = {}
    num_steps = int(np.round(beta_target/dbeta))
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_hz_%1.1f_hx_%1.5f_J_%1.1f_Ds_%s_%s" % (lattice, dims[0], dims[1], boundary, hz, hx, J, D, sym)
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
        np.save("METTS_Ising_tensors_target_beta_%1.2f_%s.npy" % (beta_target, file_name), mdata)

        # calculate observables with ctm 

        tol = 1e-10 # truncation of singular values of CTM projectors
        max_sweeps=100  # ctm param
        tol_exp = 1e-5
        opts_svd_ctm = {'D_total': chi, 'tol': tol}
        cf_energy_old = 0
        ops = {'zz': {'l': sz, 'r': sz}}

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 3rd step as iteration_step=3
            
            exp_sx, _ = one_site_avg(psi, step.env, sx) # first entry of the function gives average of one-site observables of the sites

            obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

            kin_hor = abs(obs_hor.get('zz'))
            kin_ver = abs(obs_ver.get('zz'))

            cf_energy =  - 2 * (kin_hor + kin_ver) - hx * exp_sx

            print("Energy : ", cf_energy)
            if abs(cf_energy - cf_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            cf_energy_old = cf_energy

        metts_correlator1 = EV2ptcorr(psi, step.env, ops['zz'], site0=(1,1), site1=(1,2))
        metts_correlator2 = EV2ptcorr(psi, step.env, ops['zz'], site0=(1,1), site1=(2,1))


        with open("energy_from_METTS_spinless_target_beta_%1.2f_%s.txt" % (beta_target,file_name), "a+") as f:
                f.write('{:.0f} {:.5f}\n'.format(itr+1, cf_energy))

        with open("correlator_from_METTS_spinless_target_beta_%1.2f_%s.txt" % (beta_target,file_name), "a+") as f:
                f.write('{:.0f} {:.5f} {:.5f}\n'.format(itr+1, metts_correlator1[0], metts_correlator2[0]))

        # now we do probabilistic sampling

        up, dn = sup, sdn
        projectors = [up, dn]
        out = sample(psi, step.env, projectors)
        print('out: ',out)
        psi = initialize_post_sampling_Ising(sid, sz, net, out)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=5)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=5)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='True') # purifciation can be 'True' or 'False' or 'Time'; 'True' in case of METTS
    parser.add_argument("-D", type=int, default=3)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yastn.fpeps.operators.import_distribution
    parser.add_argument("-S", default='dense')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-J", type=float, default=1.0)      # chemical potential 
    parser.add_argument("-HX", type=float, default=2.9)          # tunelling strength
    parser.add_argument("-HZ", type=float, default=0.0)          # tunelling strength
    parser.add_argument("-BT", type=float, default=0.1)        # target inverse temperature beta
    parser.add_argument("-DBETA", type=float, default=0.01)      # dbeta
    parser.add_argument("-X", type=int, default=10)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-ITER", type=int, default=500)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-STEP", default='two-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    NTU_Ising_METTS(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, sym=args.S, dbeta=args.DBETA,
                                J=args.J, beta_target=args.BT, hx=args.HX, hz=args.HZ, num_iter=args.ITER, purification=args.p,
                                chi=args.X, step=args.STEP, tr_mode=args.MODE)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 26-27 nohup python -u NTU_Ising_METTS.py -L 'rectangle' -B 'finite' -p 'True' -x 3 -y 3 -D 3 -X 15 -S 'dense' -J 1.0 -HX 2.9 -HZ 0.0 -DBETA 0.01 -BT 1.65 -ITER 5000 -STEP 'two-step' -MODE 'optimal' > METTS_Ising_finite_3_3_J_1_HX_2p9_BETA_1p65.out &
