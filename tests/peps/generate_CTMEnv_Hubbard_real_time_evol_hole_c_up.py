import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.ctm import nn_avg, ctmrg, one_site_avg
try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def ctm_env_real_time_c_up(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_end, beta_start, U, mu_up, mu_dn, t_up, t_dn, step, tr_mode, fix_bd):
    
    dims = (xx, yy)
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = (fid*1e-30+n_up) @ (fid*1e-30+n_dn)
    hole_density = (fid-n_up) @ (fid-n_dn)

    file_name = "c_up_shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("hole_initialized_real_time_evolution_Hubbard_model_spinful_tensors_%s.npy" % (file_name), allow_pickle=True).item()

    beta_range = np.arange(beta_start, beta_end, interval)
    print(beta_range)

    imb = np.zeros((len(beta_range), 5))

    tol = 1e-10 # truncation of singular values of CTM projectors
    max_sweeps=100  # ctm param
    tol_exp = 1e-6 
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    x_target = round((xx-1)*0.5)
    y_target = round((yy-1)*0.5)
    target_site = (x_target, y_target)

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}
    
    dbeta=0.005

    s= 0
    mat = {}

    for beta in beta_range:
        
        sv_beta = int((round(beta/dbeta)-1) * yastn.BETA_MULTIPLIER)
        print(beta)
    
        psi = peps.Peps(lattice, dims, boundary)
        for sind in psi.sites():
            psi[sind] = yastn.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) 
        print('BETA: ', beta)

        ctm_energy_old = 0
        for step in ctmrg(psi, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm, flag='hole'):

            assert step.sweeps % 1 == 0 # stop every 3rd step as iteration_step=3
            """with open("ctm_time_sweep_beta_%1.1f_chi_%1.1f_%s.txt" % (beta, chi, file_name),"a+") as fs:
                fs.write("{} {}\n".format(step.sweeps, step.tt))"""
              
            mean_int, _ = one_site_avg(psi, step.env, n_int, flag='hole') # first entry of the function gives average of one-site observables of the sites
            obs_hor, obs_ver =  nn_avg(psi, step.env, ops, flag='hole')

            kin_hor = abs(obs_hor.get('cdagc_up'))+ abs(obs_hor.get('ccdag_up'))+abs(obs_hor.get('cdagc_dn'))+abs(obs_hor.get('ccdag_dn'))
            kin_ver = abs(obs_ver.get('cdagc_up'))+ abs(obs_ver.get('ccdag_up'))+abs(obs_ver.get('cdagc_dn'))+abs(obs_ver.get('ccdag_dn'))

            ctm_energy = U * mean_int * (xx + yy)  - (kin_hor * (yy-1) * xx  + kin_ver * (xx-1) * yy) 
           
            print("expectation value: ", ctm_energy)
            if abs(ctm_energy - ctm_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            ctm_energy_old = ctm_energy

        mdata={}                                                                                 
        for ms in psi.sites():
            xm = {('cortl', ms, sv_beta): step.env[ms].tl.save_to_dict(), ('cortr', ms, sv_beta): step.env[ms].tr.save_to_dict(),
            ('corbl', ms, sv_beta): step.env[ms].bl.save_to_dict(), ('corbr', ms, sv_beta): step.env[ms].br.save_to_dict(),
            ('strt', ms, sv_beta): step.env[ms].t.save_to_dict(), ('strb', ms, sv_beta): step.env[ms].b.save_to_dict(),
            ('strl', ms, sv_beta): step.env[ms].l.save_to_dict(), ('strr', ms, sv_beta): step.env[ms].r.save_to_dict()}
            mdata.update(xm)
  
        with open("ctm_environment_real_time_hole_beta_%1.3f_chi_%1.1f_%s.npy" % (beta, chi, file_name), 'wb') as f:
            np.save(f, mdata, allow_pickle=True)

        print('hole_density')
        hole_density_avg, mat_hole = one_site_avg(psi, step.env, hole_density, flag='hole')
        print(mat_hole)

        print('spin polarization up')
        up_avg, mat_up = one_site_avg(psi, step.env, fid*1e-30+n_up, flag='hole')
        print(mat_up)

        print('spin polarization down')
        down_avg, mat_dn = one_site_avg(psi, step.env, fid*1e-30+n_dn, flag='hole')
        print(mat_dn)

        print('double occupancy')
        double_occupancy_avg, mat_do = one_site_avg(psi, step.env, n_int, flag='hole')
        print(mat_do)

        imb[s,0] = beta
        imb[s,1] = mat_hole[target_site]
        imb[s,2] = mat_up[target_site]
        imb[s,3] = mat_dn[target_site]
        imb[s,4] = mat_do[target_site]

        x = {('hole', sv_beta): mat_hole, ('up', sv_beta):mat_up, ('dn', sv_beta):mat_dn, ('do', sv_beta):mat_do}
        mat.update(x)
        s = s+1

    np.savetxt("spinful_chi_%1.0f_%s.txt" % (chi, file_name), imb, fmt='%.6f')
    np.save("lattice_view_hole_initialized_real_time_evolution_Hubbard_model_spinful_tensors_%s.npy" % (file_name), mat)

if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=9)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=9)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='time') # bool
    parser.add_argument("-D", type=int, default=12) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1xZ2') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=int, default=50) # chi_multiple
    parser.add_argument("-I", type=float, default=0.1) # interval
    parser.add_argument("-BETA_START", type=float, default=0.1) # location
    parser.add_argument("-BETA_END", type=float, default=1.2) # location
    parser.add_argument("-U", type=float, default=12.)       # location                                                                                             
    parser.add_argument("-MU_UP", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-MU_DOWN", type=float, default=0.) # location                                                                                           
    parser.add_argument("-T_UP", type=float, default=1.)    # location
    parser.add_argument("-T_DOWN", type=float, default=1.)  # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    ctm_env_real_time_c_up(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, U=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, t_up = args.T_UP, t_dn = args.T_DOWN, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 0-6 nohup python -u generate_CTMEnv_Hubbard_real_time_evol_c_up.py -L 'rectangle' -x 9 -y 9 -B 'finite' -p 'time' -D 12 -S 'U1xU1xZ2' -X 54 -I 0.1 -BETA_START 0.1 -BETA_END 1.2 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > up_env_real_time_spinful_9_9_gs_U_12_D_12_MU_0_T_1.out &
