import logging
import h5py
import numpy as np
import yamps.mps as mps
import transport_vectorization as main_fun
import transport_vectorization_full as main_full
import transport_vectorization_Z2 as main_Z2
import transport_vectorization_U1 as main_U1
import transport_vectorization_general as general


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("models/transport_open/transport.log"),
        #logging.StreamHandler()
    ])


def transport(main, basis, tensor):
    # main - module to produce hamiltonian 
    # choice - basis, Majorana or Dirac
    # tensor - tensor type: full, Z2 or U1
    #
    # MODEL
    NL = 3
    NS = 1
    w0 = 1
    v = .5
    wS = 0
    mu = .5
    dV = .0
    gamma = 1.  # dissipation rate
    temp = 0.025  # temperature
    distribution = 0
    # how to arrange modes. 
    # 0-spatial 1d to energy modes via sine-transformation.
    # 1-uniform spacing through W=4*w0. Are not places on the very corners of W.
    ordered = True  # order basis for mps/mpo

    # MPS
    tensor_type = main.get_tensor_type(basis)
    tol_svd = 1e-6
    D_total = 16
    io = [.5]  # initial occupation on the impurity

    # algorithm
    dt = .125 # time step - time step for single tdvp
    tmax = 20. # total time
    opts_svd = {'tol': tol_svd, 'D_total': D_total}
    eigs_tol = 1e-12

    # STORAGE
    directory = 'models/transport_open/'
    name = directory+'test_full'
    name_txt = name+'_output.txt'

    # SAVE information about simulation
    names = ['NL', 'w0', 'v', 'wS', 'mu', 'dV', 'gamma', 'temp', 'distribution',
             'ordered', 'basis', 'tol_svd', 'D_total', 'io', 'sym', 'dtype', 'dt']
    values = [NL, w0, v, wS, mu, dV, gamma, temp, distribution,
              ordered, basis, tol_svd, D_total, io, tensor_type[0], tensor_type[1], dt]
    with h5py.File(name + '_output.h5', 'w') as f:
        g_param = f.create_group('parameters')
        for inm, ival in zip(names, values):
            g_param.create_dataset(inm, data=ival)

    # EXECUTE
    LSR, wk, temp, vk, dV, gamma = general.generate_discretization(NL=NL, w0=w0, wS=wS, mu=mu, v=v, dV=dV, tempL=temp, tempR=temp, method=distribution, ordered=ordered, gamma=gamma)
    psi = main.thermal_state(tensor_type=tensor_type, LSR=LSR, io=io, ww=(wk+dV), temp=temp, basis=basis)
    LL, LdagL = main.Lindbladian_1AIM_mixed(tensor_type=tensor_type, NL=NL, LSR=LSR, wk=wk, temp=temp, vk=vk, dV=dV, gamma=gamma, basis=basis, AdagA=True)

    # TDVP
    H, hermitian, dmrg, version, HH, sgn = LL, False, False, None, LL, +1.
    #H, hermitian, dmrg, version, HH, sgn = LdagL, True, False, None, LdagL, -1.
    # DMRG 
    #H, hermitian, dmrg, version, HH = LL, False, True, None, LL
    #H, hermitian, dmrg, version, HH = LdagL, True, True, None, LdagL

    # canonize MPS
    psi.canonize_sweep(to='last')
    psi.canonize_sweep(to='first')

    # compress MPO
    H.canonize_sweep(to='last', normalize=False)
    H.sweep_truncate(to='first', opts={'tol': 1e-12}, normalize=False)
    H.canonize_sweep(to='last', normalize=False)
    # trace rho
    trace_rho = main.identity(tensor_type=tensor_type, N=psi.N, basis=basis)

    # current
    JLS = main.current(tensor_type=tensor_type, LSR=LSR, vk=-4.*np.pi*vk, cut='LS', basis=basis)
    JSR = main.current(tensor_type=tensor_type, LSR=LSR, vk=-4.*np.pi*vk, cut='SR', basis=basis)

    # Occupation
    NL = main.measure_sumOp(tensor_type=tensor_type, choice=-1, LSR=LSR, Op='nn', basis=basis)
    NS = main.measure_sumOp(tensor_type=tensor_type, choice=2, LSR=LSR, Op='nn', basis=basis)
    NR = main.measure_sumOp(tensor_type=tensor_type, choice=1, LSR=LSR, Op='nn', basis=basis)
    OP_Nocc = [0.]*psi.N
    for n in range(psi.N):
        OP_Nocc[n] = main.measure_Op(tensor_type=tensor_type, N=psi.N, id=n, Op='nn', basis=basis)

    # EVOLUTION
    env = None
    init_steps = [5,5,4,3,2,1]
    qt = 0
    it_step = 0
    qt = 0

    max_steps = int((abs(tmax)/abs(dt)).real)+10
    Es = np.zeros(max_steps)
    occ_L = np.zeros(max_steps)
    occ_S = np.zeros(max_steps)
    occ_R = np.zeros(max_steps)
    curr_LS = np.zeros(max_steps)
    curr_SR = np.zeros(max_steps)
    Ds = np.zeros((max_steps, psi.N+1))
    occ = np.zeros((max_steps, psi.N))
    # MEASURE
    Dmax = max(psi.get_D())
    E, _ = general.measure_MPOs(psi, [HH])
    E = E.real
    out, norm = general.measure_overlaps(psi, [NL, NS, NR, JLS, JSR], norm=trace_rho)
    E, nl, ns, nr, jls, jsr = (E[0]), out[0].real, out[1].real, out[2].real, out[3].real, out[4].real
    Nocc, _ = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
    Nocc=Nocc.real
    # SAVE
    Es[it_step] = E
    occ_L[it_step] = nl
    occ_S[it_step] = ns
    occ_R[it_step] = nr
    curr_LS[it_step] = jls
    curr_SR[it_step] = jsr
    Ds[it_step,:] = psi.get_D()
    occ[it_step,:] = Nocc

    # PRINT
    print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' Tot_Occ= ', round(nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
    with open(name_txt, 'w') as f:
        print(qt, 1., E, jls, jsr, nl, ns, nr, file=f)
    # EXPORT
    export = {'Es': Es, 'occ_L': occ_L, 'occ_S': occ_S, 'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
    with h5py.File(name + '_output.h5', 'w') as f:
        f.create_group('measurements')
        for inm, ival in export.items():
            f['measurements'].create_dataset(inm, data=ival)
        main_fun.save_psi_to_h5py(f, psi)
    #
    while abs(qt) < abs(tmax):
        #
        if it_step < len(init_steps) and it_step%25 < 3:
            version = '2site_group'
        else:
            version = '1site'
        if 1==0:
            if it_step%100 < 1:
                dmrg=True
            else:
                dmrg = False
        #
        if dmrg:
            ddt = dt
            _, _, _ = mps.dmrg.dmrg_OBC(psi=psi, H=LdagL, version=version, cutoff_sweep=1, eigs_tol=eigs_tol, hermitian=hermitian,  opts_svd=opts_svd)
        else:
            if it_step<len(init_steps):
                exp_tol = 1e-14 # REported to be unstable for exp_tol lower than 1e-9
                eigs_tol = 1e-14
                ddt = dt*2**(-init_steps[it_step])
            else:
                exp_tol = 1e-8
                eigs_tol = 1e-10
                ddt = dt
            env, _, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=sgn*ddt, dt=sgn*ddt, H=H, env=env, version=version, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=tensor_type[1], hermitian=hermitian,  opts_svd=opts_svd)
        #for n in range(psi.N):
        #    print(n, ':', psi.A[n].to_numpy(),'\n')
        qt += abs(ddt)
        # MEASURE
        Dmax = max(psi.get_D())
        E, _ = general.measure_MPOs(psi, [HH])
        E = E.real
        out, norm = general.measure_overlaps(psi, [NL, NS, NR, JLS, JSR], norm=trace_rho)
        E, nl, ns, nr, jls, jsr = (E[0]), out[0].real, out[1].real, out[2].real, out[3].real, out[4].real
        Nocc, _ = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
        Nocc=Nocc.real
        # SAVE
        Es[it_step] = E
        occ_L[it_step] = nl
        occ_S[it_step] = ns
        occ_R[it_step] = nr
        curr_LS[it_step] = jls
        curr_SR[it_step] = jsr
        Ds[it_step,:] = psi.get_D()
        occ[it_step,:] = Nocc
        # PRINT
        print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' N = ', norm, ' Tot_Occ= ', round(nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
        with open(name_txt, 'a') as f:
            print(qt, Dmax, E, jls, jsr, nl, ns, nr, file=f)
        # EXPORT
        export = {'Es': Es, 'occ_L': occ_L, 'occ_S': occ_S, 'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
        if it_step%100 == 0:
            with h5py.File(name + '_output.h5', 'w') as f:
                f.create_group('measurements')
                for inm, ival in export.items():
                    f['measurements'].create_dataset(inm, data=ival)
                main_fun.save_psi_to_h5py(f, psi)
        it_step += 1
    

def transport_full(choice):
    transport(main_full, choice, 'full')


def transport_Z2(choice):
    if choice == 'Majorana':
        transport(main_Z2, choice, 'Z2')
    elif choice == 'Dirac':
        print('Z2/Dirac - Option is not implemented.')
        pass


def transport_U1(choice):
    if choice == 'Dirac':
        transport(main_U1, choice, 'U1')
    elif choice == 'Majorana':
        print('U1/Majorana - Option is not implemented.')
        pass


if __name__ == "__main__":
    # pass
    transport_full('Majorana'); print()
    #transport_full('Dirac'); print()
    #transport_Z2('Majorana'); print()
    #transport_Z2('Dirac'); print()
    #transport_U1('Majorana'); print()
    #transport_U1('Dirac'); print()
