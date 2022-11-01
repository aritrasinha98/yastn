""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from ... import ncon, tensordot, expmv, vdot, qr, svd, ones
from ._mps import YampsError


def measure_overlap(bra, ket):
    r"""
    Calculate overlap :math:`\langle \textrm{bra}|\textrm{ket} \rangle`.
    Conjugate of MPS :code:`bra` is computed internally.
    
    MPSs :code:`bra` and :code:`ket` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra : yamps.MpsMpo
        An MPS which will be conjugated.

    ket : yamps.MpsMpo

    Returns
    -------
    scalar
    """
    env = Env2(bra=bra, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op, ket):
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.
    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra : yamps.MpsMpo
        An MPS which will be conjugated.

    op : yamps.MpsMpo
        Operator written as MPO.

    ket : yamps.MpsMpo

    Returns
    -------
    scalar
    """
    env = Env3(bra=bra, op=op, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


class _EnvParent:
    def __init__(self, bra=None, ket=None, project=None) -> None:
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.N = ket.N
        self.nr_phys = ket.nr_phys
        self.nr_layers = 2
        self.F = {}  # dict for environments
        self.ort = [] if project is None else project
        self.Fort = [{} for _ in range(len(self.ort))]
        self._temp = {}
        self.reset_temp()

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YampsError('bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YampsError('bra and ket should have the same number of sites.')

        config = self.ket[0].config
        for ii in range(len(self.ort)):
            legs = [self.ort[ii].get_leftmost_leg(), self.ket.get_leftmost_leg().conj()]
            self.Fort[ii][(-1, 0)] = ones(config=config, legs=legs)
            legs = [self.ket.get_rightmost_leg().conj(), self.ort[ii].get_rightmost_leg()]
            self.Fort[ii][(self.N, self.N - 1)] = ones(config=config, legs=legs)  # TODO: eye ?

    def reset_temp(self):
        """ Reset temporary objects stored to speed-up some simulations. """
        self._temp = {'Aort': [], 'op_2site': {}, 'expmv_ncv': {}}

    def setup(self, to='last'):
        r"""
        Setup all environments in the direction given by to.

        Parameters
        ----------
        to : str
            'first' or 'last'.
        """
        for n in self.ket.sweep(to=to):
            self.update_env(n, to=to)
        return self

    def clear_site(self, *args):
        r""" Clear environments pointing from sites which indices are provided in args. """
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at bd bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.

        Returns
        -------
        overlap : float or complex
        """
        if bd is None:
            bd = (-1, 0)
        axes = ((0, 1), (1, 0)) if self.nr_layers == 2 else ((0, 1, 2), (2, 1, 0))
        return tensordot(self.F[bd], self.F[bd[::-1]], axes=axes).to_number()

    def project_ket_on_bra(self, n):
        r"""
        Project ket on a n-th site of bra.

        It is equal to the overlap <bra|op|ket> up to the contribution from n-th site of bra.

        Parameters
        ----------
        n : int
            index of site

        Returns
        -------
        out : tensor
            result of projection
        """
        return self.Heff1(self.ket[n], n)

    def update_env(self, n, to='last'):
        r"""
        Update environment including site n, in the direction given by to.

        Parameters
        ----------
        n: int
            index of site to include to the environment

        to : str
            'first' or 'last'.
        """
        if self.nr_layers == 2:
            _update2(n, self.F, self.bra, self.ket, to, self.nr_phys)
        else:
            _update3(n, self.F, self.bra, self.op, self.ket, to, self.nr_phys, self.on_aux)
        for ii in range(len(self.ort)):
            _update2(n, self.Fort[ii], self.bra, self.ort[ii], to, self.nr_phys)

    def update_Aort(self, n):
        """ Update projection of states to be subtracted from psi. """
        Aort = []
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        for ii in range(len(self.ort)):
            Aort.append(ncon([self.Fort[ii][(n - 1, n)], self.ort[ii][n], self.Fort[ii][(n + 1, n)]], inds))
        self._temp['Aort'] = Aort

    def update_AAort(self, bd):
        """ Update projection of states to be subtracted from psi. """
        Aort = []
        nl, nr = bd
        inds = ((-0, 1), (1, -1, -2,  2), (2, -3)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        for ii in range(len(self.ort)):
            AA = self.ort[ii].merge_two_sites(bd)
            Aort.append(ncon([self.Fort[ii][(nl - 1, nl)], AA, self.Fort[ii][(nr + 1, nr)]], inds))
        self._temp['Aort'] = Aort

    def _project_ort(self, A):
        for ii in range(len(self.ort)):
            x = vdot(self._temp['Aort'][ii], A)
            A = A.apxb(self._temp['Aort'][ii], -x)
        return A


class Env2(_EnvParent):
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, ket=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket)

        # left boundary
        config = self.bra[0].config
        legs = [self.bra.get_leftmost_leg(), self.ket.get_leftmost_leg().conj()]
        self.F[(-1, 0)] = ones(config=config, legs=legs)  # TODO: or eye?
        # right boundary
        legs = [self.ket.get_rightmost_leg().conj(), self.bra.get_rightmost_leg()]
        self.F[(self.N, self.N - 1)] = ones(config=config, legs=legs)  # TODO: or eye?

    def Heff1(self, x, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A : tensor
            site tensor

        n : int
            index of corresponding site

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        return ncon([self.F[(n - 1, n)], x, self.F[(n + 1, n)]], inds)


class Env3(_EnvParent):
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, op=None, ket=None, on_aux=False, project=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm opp} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        opp : mps
            mps for operator opp.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket, project)
        self.op = op
        self.nr_layers = 3
        self.on_aux = on_aux
        if self.op.N != self.N:
            raise YampsError('op should should have the same number of sites as ket.')

        # left boundary
        config = self.ket[0].config
        legs = [self.bra.get_leftmost_leg(), self.op.get_leftmost_leg().conj(), self.ket.get_leftmost_leg().conj()]
        self.F[(-1, 0)] = ones(config=config, legs=legs)

        # right boundary
        legs = [self.ket.get_rightmost_leg().conj(), self.op.get_rightmost_leg().conj(), self.bra.get_rightmost_leg()]
        self.F[(self.N, self.N - 1)] = ones(config=config, legs=legs)

    def Heff0(self, C, bd):
        r"""
        Action of Heff on central site.

        Parameters
        ----------
        C : tensor
            a central site
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff0 * C
        """
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        return ncon([self.F[bd], C, self.F[ibd]], ((-0, 2, 1), (1, 3), (3, 2, -1)))


    def Heff1(self, A, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A : tensor
            site tensor

        n : int
            index of corresponding site

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        nl, nr = n - 1, n + 1
        A = self._project_ort(A)

        if self.nr_phys == 1:
            tmp = A @ self.F[(nr, n)]
            tmp = self.op[n]._attach_23(tmp)
            tmp = ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
        elif self.nr_phys == 2 and not self.on_aux:
            tmp = A.fuse_legs(axes=((0, 3), 1, 2))
            tmp = ncon([tmp, self.F[(nr, n)]], ((-0, -1, 1), (1, -2, -3)))
            tmp = self.op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            tmp = ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, -3, 1, -2, -1)))
        else:  # if self.nr_phys == 2 and self.on_aux:
            tmp = ncon([A, self.F[(nl, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
            tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
            tmp = self.op[n]._attach_01(tmp)
            tmp = ncon([tmp, self.F[(nr, n)]], ((1, 2, -0, -2), (1, 2, -1)))
            tmp = tmp.unfuse_legs(axes=0)

        return self._project_ort(tmp)


    # def Heff2(self, AA, bd):
    #     r"""Action of Heff on central site.

    #     Parameters
    #     ----------
    #     AA : tensor
    #         merged tensor for 2 sites.
    #         Physical legs should be fused turning it effectivly into 1-site update.
    #     bd : tuple
    #         index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

    #     Returns
    #     -------
    #     out : tensor
    #         Heff2 * AA
    #     """
    #     n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
    #     bd, nl, nr = (n1, n2), n1 - 1, n2 + 1

    #     if bd not in self._temp['op_2site']:
    #         OO = tensordot(self.op[n1], self.op[n2], axes=(2, 0))
    #         self._temp['op_2site'][bd] = OO.fuse_legs(axes=(0, (1, 3), 4, (2, 5)))
    #     OO = self._temp['op_2site'][bd]

    #     AA = self._project_ort(AA)
    #     if self.nr_phys == 1:
    #         return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
    #                     ((-0, 2, 1), (1, 3, 4), (2, -1, 5, 3), (4, 5, -2)))
    #     if self.nr_phys == 2 and not self.on_aux:
    #         return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
    #                     ((-0, 2, 1), (1, 3, 4, -3), (2, -1, 5, 3), (4, 5, -2)))
    #     return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
    #                     ((-0, 2, 1), (1, -1, 4, 3), (2, -3, 5, 3), (4, 5, -2)))


    def Heff2(self, AA, bd):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectivly into 1-site update.
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1

        tmp = self._project_ort(AA)
        if self.nr_phys == 1:
            tmp = tmp @ self.F[(nr, n2)]
            tmp = tmp.fuse_legs(axes=((0, 1), 2, 3, 4))
            tmp = self.op[n2]._attach_23(tmp)
            tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
            tmp = tmp.unfuse_legs(axes=0)
            tmp = self.op[n1]._attach_23(tmp)
            tmp = ncon([self.F[(nl, n1)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
            tmp = tmp.unfuse_legs(axes=2)
        # if self.nr_phys == 2 and not self.on_aux:
        #     return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
        #                 ((-0, 2, 1), (1, 3, 4, -3), (2, -1, 5, 3), (4, 5, -2)))
        return self._project_ort(tmp)

    def update_A(self, n, du, opts, normalize=True):
        """ Updates env.ket[n] by exp(du Heff1). """
        if n in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][n]
        f = lambda x: self.Heff1(x, n)
        self.ket[n], info = expmv(f, self.ket[n], du, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][n] = info['ncv']

    def update_C(self, du, opts, normalize=True):
        """ Updates env.ket[bd] by exp(du Heff0). """
        bd = self.ket.pC
        if bd[0] != -1 and bd[1] != self.N:  # do not update central sites outsite of the chain
            if bd in self._temp['expmv_ncv']:
                opts['ncv'] = self._temp['expmv_ncv'][bd]
            f = lambda x: self.Heff0(x, bd)
            self.ket.A[bd], info = expmv(f, self.ket[bd], du, **opts, normalize=normalize, return_info=True)
            self._temp['expmv_ncv'][bd] = info['ncv']

    def update_AA(self, bd, du, opts, opts_svd, normalize=True):
        """ Merge two sites given in bd into AA, updates AA by exp(du Heff2) and unmerge the sites. """
        ibd = bd[::-1]
        if ibd in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][ibd]
        AA = self.ket.merge_two_sites(bd)
        f = lambda v: self.Heff2(v, bd)
        AA, info = expmv(f, AA, du, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][ibd] = info['ncv']
        self.ket.unmerge_two_sites(AA, bd, opts_svd)

    def enlarge_bond(self, bd, opts_svd):
        if bd[0] < 0 or bd[1] >= self.N:  # do not enlarge bond outside of the chain
            return False
        AL = self.ket[bd[0]]
        AR = self.ket[bd[1]]
        if self.op[bd[0]].get_legs(axis=1).t != AL.get_legs(axis=1).t or \
           self.op[bd[1]].get_legs(axis=1).t != AR.get_legs(axis=1).t:
            return True  # true if some charges are missing on physical legs of psi

        AL = AL.fuse_legs(axes=((0, 1), 2))
        AR = AR.fuse_legs(axes=(0, (1, 2)))
        shapeL = AL.get_shape()
        shapeR = AR.get_shape()
        if shapeL[0] == shapeL[1] or shapeR[0] == shapeR[1] or \
           ('D_total' in opts_svd and shapeL[0] >= opts_svd['D_total']):
            return False  # maximal bond dimension
        if 'tol' in opts_svd:
            _, R0 = qr(AL, axes=(0, 1), sQ=-1)
            _, R1 = qr(AR, axes=(1, 0), Raxis=1, sQ=1)
            _, S, _ = svd(R0 @ R1)
            if any(S[t][-1] > opts_svd['tol'] * 1.1 for t in S.struct.t):
                return True
        return False


def _update2(n, F, bra, ket, to, nr_phys):
    """ Contractions for 2-layer environment update. """
    if to == 'first':
        inds = ((-0, 2, 1), (1, 3), (-1, 2, 3)) if nr_phys == 1 else ((-0, 2, 1, 4), (1, 3), (-1, 2, 3, 4))
        F[(n, n - 1)] = ncon([ket[n], F[(n + 1, n)], bra[n].conj()], inds)
    elif to == 'last':
        inds = ((2, 3, -0), (2, 1), (1, 3, -1)) if nr_phys == 1 else ((2, 3, -0, 4), (2, 1), (1, 3, -1, 4))
        F[(n, n + 1)] = ncon([bra[n].conj(), F[(n - 1, n)], ket[n]], inds)


def _update3(n, F, bra, op, ket, to, nr_phys, on_aux):
    if nr_phys == 1 and to == 'last':
        tmp = ncon([bra[n].conj(), F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
        tmp = op[n]._attach_01(tmp)
        F[(n, n + 1)] = ncon([tmp, ket[n]], ((-0, -1, 1, 2), (1, 2, -2)))
    elif nr_phys == 1 and to == 'first':
        tmp = ket[n] @ F[(n + 1, n)]
        tmp = op[n]._attach_23(tmp)
        F[(n, n - 1)] = ncon([tmp, bra[n].conj()], ((-0, -1, 1, 2), (-2, 2, 1)))
    elif nr_phys == 2 and not on_aux and to == 'last':
        bA = bra[n].fuse_legs(axes=(0, 1, (2, 3)))
        tmp = ncon([bA.conj(), F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
        tmp = op[n]._attach_01(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n + 1)] = ncon([tmp, ket[n]], ((-0, 3, -1, 1, 2), (1, 2, -2, 3)))
    elif nr_phys == 2 and not on_aux and to == 'first':
        kA = ket[n].fuse_legs(axes=((0, 3), 1, 2))
        tmp = ncon([kA, F[(n + 1, n)]], ((-0, -1, 1), (1, -2, -3)))
        tmp = op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n - 1)] = ncon([tmp, bra[n].conj()], ((-0, 3, -1, 1, 2), (-2, 2, 1, 3)))
    elif nr_phys == 2 and on_aux and to == 'last':
        tmp = ncon([ket[n], F[(n - 1, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
        tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
        tmp = op[n]._attach_01(tmp)
        bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        F[(n, n + 1)] = ncon([bA.conj(), tmp], ((1, -0, 2), (-2, -1, 1, 2)))
    else: # nr_phys == 2 and on_aux and to == 'first':
        bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        tmp = ncon([bA.conj(), F[(n + 1, n)]], ((-0, 1, -1), (-3, -2, 1)))
        tmp = op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n - 1)] = ncon([ket[n], tmp], ((-0, 1, 2, 3), (-2, 1, -1, 2, 3)))
