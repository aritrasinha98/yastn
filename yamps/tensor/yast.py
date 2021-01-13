r"""
Yet another symmetric tensor

This class defines generic arbitrary rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used (including no symmetries).

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""

import logging
import numpy as np
import itertools

logger = logging.getLogger('yast')

class FatalError(Exception):
    pass

#######################################################
#     Functions creating and filling in new tensor    #
#######################################################

def randR(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with real random numbers in [-1, 1].

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag)
    a.fill_tensor(t=t, D=D, val='randR')
    return a


def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers in [-1, 1] and type specified in config.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag)
    a.fill_tensor(t=t, D=D, val='rand')
    return a


def zeros(config=None, s=(), n=None, t=(), D=(), **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with zeros
    """
    a = Tensor(config=config, s=s, n=n, isdiag=False)
    a.fill_tensor(t=t, D=D, val='zeros')
    return a


def ones(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with ones
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def eye(config=None, t=(), D=(), **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of diagonal tensor filled with ones
    """
    a = Tensor(config=config, isdiag=True)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def from_dict(config=None, d={'s': (), 'n': None, 'isdiag': False, 'A': {}}):
    """
    Generate tensor based on information in dictionary d.

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    a = Tensor(config=config, s=d['s'], n=d['n'], isdiag=d['isdiag'])
    for ind in d['A']:
        a.set_block(ts=ind, Ds=d['A'][ind].shape, val=d['A'][ind])
    return a


def decompress_from_1d(r1d, config=None, d={}):
    """
    Generate tensor based on information in dictionary d and 1D array
    r1d containing the serialized blocks

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    a = Tensor(config=config, s=d['s'], n=d['n'], isdiag=d['isdiag'])
    A = {(): r1d}
    a.A = a.config.backend.unmerge_one_leg(A, 0, 1000, d['meta_umrg'])
    a.lss = {t:ls.copy() for t, ls in d['lss'].items()}
    a._calculate_tDset()
    return a


def match_legs(tensors=None, legs=None, conjs=None, val='ones', isdiag=False):
    r"""
    Initialize tensor matching legs of existing tensors, so that it can be contracted with those tensors.

    Finds all matching symmetry sectors and their bond dimensions and passes it to :meth:`Tensor.fill_tensor`.
    Can creat diagonal tensor by matching to one leg of one other tensor.

    Parameters
    ----------
    tensors: list
        list of tensors -- they should not be diagonal to properly identify signature.
    legs: list
        and their corresponding legs to match
    conjs: list
        if tensors are entering dot as conjugated
    val: str
        'randR', 'rand', 'ones', 'zeros'
    """
    t, D, s = [], [], []
    if conjs is None:
        conjs = [0] * len(tensors)
    for n, te, cc in zip(legs, tensors, conjs):
        tdn = te.get_leg_tD(n)
        t.append(tuple(tdn.keys()))
        D.append(tuple(tdn.values()))
        s.append(te.s[n] * (2 * cc - 1))
    a = tensors[0].empty(s=s, isdiag=isdiag)
    a.fill_tensor(t=t, D=D, val=val)
    return a


def block(td, common_legs):
    """ Assemble new tensor by blocking a set of tensors.

        Parameters
        ----------
        td : dict
            dictionary of tensors {(k,l): tensor at position k,l}.
            Length of tuple should be equall to tensor.ndim - len(common_legs)

        common_legs : list
            Legs which are not blocked

        ndim : int
            All tensor should have the same rank ndim
    """
    try:
        ls = len(common_legs)
        out_s = tuple(common_legs)
    except TypeError:
        out_s = (common_legs,)
        ls = 1

    a = next(iter(td.values()))  # first tensor, used to initialize and retrive common values
    ndim = a.ndim
    nsym = a.config.sym.nsym

    out_m = tuple(ii for ii in range(ndim) if ii not in out_s)
    # out_ma = np.array(out_m, dtype=np.intp)
    li = ndim - ls
    pos = []
    for ind in td:
        if li != len(ind):
            logger.exception('block: wrong tensors rank or placement')
            raise FatalError
        pos.append(ind)
    pos.sort()

    # all charges and bond dimensions
    tlist, Dlist = {}, {}
    for ind in pos:
        tt, DD = [], []
        for n in range(ndim):
            tDleg = td[ind].get_leg_tD(n)
            tt.append(sorted([k for k in tDleg.keys()]))  
            DD.append([tDleg[k] for k in tt[-1]])
        tlist[ind] = tt
        Dlist[ind] = DD

    # all unique blocks to block
    t_out_unique = sorted(set([ind for x in td.values() for ind in x.A]))
    meta = []
    for t in t_out_unique:
        ta = np.array(t, dtype=int).reshape(ndim, nsym)
        legs_ind = []  # indices on specific legs
        legs_D = []  # and corresponding D
        kk = -1
        for ii in range(ndim):
            tl = tuple(ta[ii].flat)
            relevant_pos = []
            relevant_D = []
            for ind in pos:
                if tl in tlist[ind][ii]:
                    relevant_pos.append(ind)
                    ind_tl = tlist[ind][ii].index(tl)
                    relevant_D.append(Dlist[ind][ii][ind_tl])
            if ii in out_m:
                kk += 1
                posa = np.array(relevant_pos, dtype=int)
                x, y = np.unique(posa[:, kk], return_index=True)
                legs_ind.append(list(x))
                legs_D.append([relevant_D[ll] for ll in y])  # all D's for unique positions should be identical -- does not check this
            else:
                legs_D.append([relevant_D[0]])  # all should be identical -- does not check this
        meta.append((t, legs_ind, legs_D))

    c = Tensor(config=a.config, s=a.s, isdiag=a.isdiag)
    c.A = c.config.backend.block(td, meta, dtype=a.config.dtype)
    c._calculate_tDset()
    return c

class Tensor:
    """
    Class defining a tensor with abelian symmetries and main operations acting on such tensor(s).
    """

    def __init__(self, config=None, s=(), n=None, isdiag=False, lss=None, **kwargs):
        self.config = kwargs['settings'] if 'settings' in kwargs else config
        self.isdiag = isdiag
        self.ndim = 1 if isinstance(s, int) else len(s)  # number of legs
        self.s = np.array(s, dtype=int).reshape(self.ndim)
        self.n = np.zeros(self.config.sym.nsym, dtype=int) if n is None else np.array(n, dtype=int).reshape(self.config.sym.nsym)
        if self.isdiag:
            if len(self.s) == 0:
                self.s = np.array([1, -1], dtype=int)
                self.ndim = 2
            if not np.sum(self.s) == 0:
                logger.exception("Signature should be (-1, 1) or (1, -1) in diagonal tensor")
                raise FatalError
            if not np.sum(np.abs(self.n)) == 0:
                logger.exception("Tensor charge should be 0 in diagonal tensor")
                raise FatalError
            if not self.ndim == 2:
                logger.exception("Diagonal tensor should have ndim == 2")
                raise FatalError
        self.tset = np.zeros((0, self.ndim, self.config.sym.nsym), dtype=int)  # list of blocks; 3d nparray of ints
        self.Dset = np.zeros((0, self.ndim), dtype=int)  # shapes of blocks; 2d nparray of ints
        self.A = {}  # dictionary of blocks
        self.lss = {}  # dictionary of leg structures (of merged legs)

    def copy(self):
        r"""
        Return a copy of the tensor.
        """
        a = Tensor(config=self.config, s=self.s, n=self.n, isdiag=self.isdiag)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        a.A = {ts: self.config.backend.copy(x) for ts, x in self.A.items()}
        a.lss = {leg: ls.copy() for leg, ls in self.lss.items()}
        return a

    def detach_(self):
        self.conf.backend.detach_(self.A)

    def move_to_device(self, device):
        self.A= self.conf.backend.move_to_device(self.A, device)
        return self

    ######################
    #     fill tensor    #
    ######################

    def reset_tensor(self, t=(), D=(), val='rand', **kwargs):
        self.fill_tensor(t, D, val)
    
    def fill_tensor(self, t=(), D=(), val='rand'):
        r"""
        Create all possible blocks based on s, n and list of charges for all legs.

        Brute-force check all possibilities and select the ones satisfying f(t@s) == n for each symmetry generator f.
        Initialize each possible block with sizes given by D.

        Parameters
        ----------
        t : list
            All possible combination of charges for each leg:
            t = [[(leg1sym1, leg1sym2), ... ], [(leg2sym1, leg2sym2), ... )]
            If nsym is 0, it is not taken into account.
            When somewhere there is only one value and it is unambiguous, tuple can typically be replaced by int, see examples.
            
        D : tuple
            list of bond dimensions on all legs
            If nsym == 0, D = [leg1, leg2, leg3]
            If nsym >= 1, it should match the form of t
            When somewhere there is only one value tuple can typically be replaced by int.

        val : str
            'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'

        Examples
        --------
        D = 5 # ndim = 1)
        D = (1, 2, 3) # nsym = 0, ndim = 3
        t = [0, (-2, 0), (2, 0)] D=[1, (1, 2), (1, 3)] # nsym = 1 ndim = 3
        t = [[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]] D=[1, (1, 4, 2, 2), (1, 9, 3, 3)] # nsym = 2 ndim = 3
        """
        D = (D,) if isinstance(D, int) else D
        t = (t,) if isinstance(t, int) else t

        if self.config.sym.nsym == 0:
            if self.isdiag and len(D) == 1:
                D = D + D
            if len(D) != self.ndim:
                logger.exception("Number of elements in D does not match tensor rank.")
                raise FatalError
            tset = np.zeros((1, self.ndim, self.config.sym.nsym))
            Dset = np.array(D, dtype=int).reshape(1, self.ndim)
        else:  # self.config.sym.nsym >= 1
            D = (D,) if (self.ndim == 1 or self.isdiag) and isinstance(D[0], int) else D
            t = (t,) if (self.ndim == 1 or self.isdiag) and isinstance(t[0], int) else t
            D = D + D if self.isdiag and len(D) == 1 else D
            t = t + t if self.isdiag and len(t) == 1 else t

            D = list(x if isinstance(x, tuple) or isinstance(x, list) else (x, ) for x in D)
            t = list(x if isinstance(x, tuple) or isinstance(x, list) else (x, ) for x in t)
    
            if len(D) != self.ndim:
                logger.exception("Number of elements in D does not match tensor rank.")
                raise FatalError
            if len(t) != self.ndim:
                logger.exception("Number of elements in t does not match tensor rank.")
                raise FatalError
            for x, y in zip(D, t):
                if len(x) != len(y):
                    logger.exception("Elements of t and D do not match")
                    raise FatalError

            comb_t = list(itertools.product(*t))
            comb_D = list(itertools.product(*D))
            lcomb_t = len(comb_t)
            comb_t = np.array(comb_t, dtype=int).reshape(lcomb_t, self.ndim, self.config.sym.nsym)
            comb_D = np.array(comb_D, dtype=int).reshape(lcomb_t, self.ndim)
            ind = np.all(self.config.sym.fuse(comb_t, self.s, 1) == self.n, axis=1)
            tset = comb_t[ind]
            Dset = comb_D[ind]

        for ts, Ds in zip(tset, Dset):
            self.set_block(tuple(ts.flat), tuple(Ds), val)

    def set_block(self, ts=(), Ds=None, val='zeros'):
        """
        Add new block to tensor or change the existing one.

        This is the intended way to add new blocks by hand. 
        Checks if bond dimensions of the new block are consistent with the existing ones.
        Updates meta-data.

        Parameters
        ----------
        ts : tuple
            charges identifing the block, t = (sym1leg1, sym2leg1, sym1leg2, sym2leg2, ...)
            If nsym == 0, it is not taken into account.

        Ds : tuple
            bond dimensions of the block. Ds = (leg1, leg2, leg3)
            If Ds not given, tries to read it from existing blocks.

        val : str, nparray, list
            'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'
            for nparray setting Ds is needed.
        """
        if isinstance(Ds, int):
            Ds = (Ds,)
        if isinstance(ts, int):
            ts = (ts,)
        if self.isdiag and Ds is not None and len(Ds) == 1:
            Ds = Ds + Ds
        if self.isdiag and len(ts) == self.config.sym.nsym:
            ts = ts + ts

        if (len(ts) != self.ndim * self.config.sym.nsym):
            logger.exception('Wrong size of ts.')
            raise FatalError
        if (Ds is not None and len(Ds) != self.ndim):
            logger.exception('Wrong size of Ds.')
            raise FatalError        

        ats = np.array(ts, dtype=int).reshape(1, self.ndim, self.config.sym.nsym)
        if not np.all(self.config.sym.fuse(ats, self.s, 1) == self.n):
            logger.exception('Charges ts are not consistent with the symmetry rules: t @ s - n != 0')
            raise FatalError

        if Ds is None:
            Ds = []
            tD = [self.get_leg_tD(n) for n in range(self.ndim)]
            for n in range(self.ndim):
                try:
                    Ds.append(tD[n][tuple(ats[0, n, :].flat)])
                except KeyError:
                    logger.exception('Cannot infer all bond dimensions')
                    raise FatalError
        Ds = tuple(Ds)       

        if isinstance(val, str):
            if val == 'zeros':
                self.A[ts] = self.config.backend.zeros(Ds, dtype=self.config.dtype)
            elif val == 'randR':
                self.A[ts] = self.config.backend.randR(Ds, dtype=self.config.dtype)
            elif val == 'rand':
                self.A[ts] = self.config.backend.rand(Ds, dtype=self.config.dtype)
            elif val == 'ones':
                self.A[ts] = self.config.backend.ones(Ds, dtype=self.config.dtype)
            if self.isdiag:
                self.A[ts] = self.config.backend.diag_get(self.A[ts])
                self.A[ts] = self.config.backend.diag_create(self.A[ts])
        else:
            if self.isdiag and val.ndim == 1 and np.prod(Ds)==(val.size**2):
                self.A[ts] = self.config.backend.to_tensor(np.diag(val), Ds, dtype=self.config.dtype)
            else:
                self.A[ts] = self.config.backend.to_tensor(val, Ds, dtype=self.config.dtype)
        # here it checkes the consistency of bond dimensions
        self._calculate_tDset()
        tD = [self.get_leg_tD(n) for n in range(self.ndim)]

    #######################
    #     new tensors     #
    #######################

    def empty(self, s=None, n=None, isdiag=False, **kwargs):
        r"""
        Initialize a new tensor using the same tensor config.

        Parameters
        ----------
        s : tuple
            a signature of tensor
        n : int
            total charges in all symmetry sectors
        isdiag : bool
            makes tensor diagonal
        Returns
        -------
        tensor : Tensor
            empty tensor
        """
        return Tensor(config=self.config, s=s, n=n, isdiag=isdiag)

    def from_dict(self, **kwargs):
        r"""
        Wraper to :meth:`from_dict`, passing the tensor config.
        """
        return from_dict(config=self.config, **kwargs)

    def rand(self, **kwargs):
        r"""
        Wraper to :meth:`rand`, passing the tensor config.
        """
        return rand(config=self.config, **kwargs)

    def zeros(self, **kwargs):
        r"""
        Wraper to :meth:`zeros`, passing the tensor config.
        """
        return zeros(config=self.config, **kwargs)

    def ones(self, **kwargs):
        r"""
        Wraper to :meth:`ones`, passing the tensor config.
        """
        return ones(config=self.config, **kwargs)

    def eye(self, **kwargs):
        r"""
        Wraper to :meth:`eye`, passing the tensor config.
        """
        return eye(config=self.config, **kwargs)

    ###########################
    #    output statistics    #
    ###########################

    def show_properties(self):
        """ Display basic properties of the tensor. """
        print("ndim      :", self.ndim)
        print("signature :", self.s)
        print("charge    :", self.n)
        print("isdiag    :", self.isdiag)
        print("blocks    :", len(self.A))
        print("size      :", self.get_size())
        tDs = [self.get_leg_tD(n) for n in range(self.ndim)]
        Dtot = tuple(sum(tD.values()) for tD in tDs)
        print("total dim :", Dtot)
        for n in range(self.ndim):
            print("Leg", n, ":", tDs[n])
        print()

    def to_dict(self):
        r"""
        Export relevant information about tensor to dictionary -- so that it can be saved using numpy.save

        Returns
        -------
        d: dict
            dictionary containing all the information needed to recreate the tensor.
        """
        AA = {ind: self.config.backend.to_numpy(self.A[ind]) for ind in self.A}
        if self.isdiag:
            AA = {t: np.diag(x) for t, x in AA.items()}
        out = {'A': AA, 's': tuple(self.s), 'n': tuple(self.n), 'isdiag': self.isdiag}
        return out

    def compress_to_1d(self):
        """ 
        store each block as 1D array within r1d in contiguous manner and 
        record info about charges and dimensions in a list
        """
        D_rsh = np.prod(self.Dset, axis=1)
        aD_rsh = np.cumsum(D_rsh)
        D_tot = np.sum(D_rsh)
        meta_new = (((), D_tot),)
        # meta_mrg = ((tn, Ds, to, Do), ...)
        meta_mrg = tuple(((), (aD-D, aD), tuple(t.flat), (D,)) for t, D, aD in zip(self.tset, D_rsh, aD_rsh))
        A = self.config.backend.merge_one_leg(self.A, 0, tuple(range(self.ndim)), meta_new, meta_mrg, self.config.dtype)
        
        # (told, tnew, Dsl, Dnew)
        meta_umrg = tuple((told, tnew, Dsl, tuple(Dnew)) for (told, Dsl, tnew, _), Dnew in zip(meta_mrg, self.Dset))
        meta = {'s': tuple(self.s), 'n': tuple(self.n), 'isdiag': self.isdiag, 
                'lss':{n:ls.copy() for n, ls in self.lss.items()}, 'meta_umrg':meta_umrg}
        return meta, A[()]

    def __str__(self):
        return str(self.A)

    def get_size(self):
        """ Total number of elements in tensor. """
        return sum(np.prod(self.Dset, axis=1))

    def get_tensor_charge(self):
        """ Global charges of the tensor. """
        return tuple(self.n)

    def get_signature(self):
        """ Tensor signature. """
        return tuple(self.s)

    def get_charges(self):
        """ Charges of all blocks. """
        return self.tset.copy()

    def get_shapes(self):
        """ Shapes fo all blocks. """
        return self.Dset.copy()

    def get_leg_tD(self, n):
        """
        Find all charges tn for n-th leg and the corresponding bond dimensions Dn.
        
        Check if the bond dimensions of blocks are consistent.

        Returns
        -------
            tDn : dict
                format is {tn: Dn}
        """
        tset = self.tset[:, n, :]
        Dset = self.Dset[:, n]
        tDn = {}
        for tn, Dn in zip(tset, Dset):
            ind = tuple(tn.flat)
            if (ind in tDn) and tDn[ind] != Dn:
                logger.exception('Inconsistend bond dimension of charge.') 
                raise FatalError
            tDn[ind] = Dn
        return tDn

    def get_total_shape(self):
        """ Total bond dimension of all legs."""
        return tuple(sum(self.get_leg_tD(n).values()) for n in range(self.ndim))

    #########################
    #    output numbers     #
    #########################

    def to_dense(self, tDs=None):
        r"""
        Create full tensor corresponding to the symmetric tensor.

        Blockes are ordered according to increasing charges on each leg.
        It is possible to supply a list of charges and bond dimensions to be included
        (should be consistent with the tensor). This allows to fill in some zero blocks.

        Parameters
        ----------
        tDs : dict
            {n: {tn: Dn}} specify charges and dimensions to include on some legs.

        Returns
        -------
        out : tensor used by backend
        """
        tD = [self.get_leg_tD(n) for n in range(self.ndim)]
        if tDs is not None:
            for n, tDn in tDs.items():
                if (n<0) or (n >= self.ndim):
                    logger.exception('Specified leg out of ndim')
                    raise FatalError
                for tn, Dn in tDn.items():
                    if (tn in tD[n]) and tD[n][tn] != Dn:
                        logger.exception('Specified bond dimensions inconsistent with tensor.') 
                        raise FatalError
                    tD[n][tn] = Dn
        Dtot = [sum(tDn.values()) for tDn in tD]
        for tDn in tD:
            tns = sorted(tDn.keys())
            Dlow = 0
            for tn in tns:
                Dhigh = Dlow + tDn[tn]
                tDn[tn] = (Dlow, Dhigh)
                Dlow = Dhigh
        meta = []
        for ind in self.tset:
            tt = tuple(ind.flat)
            meta.append((tt, tuple(tD[n][tuple(tn.flat)] for n, tn in enumerate(ind))))
        Anew = self.config.backend.merge_to_dense(self.A, Dtot, meta, self.config.dtype)
        return Anew

    def to_numpy(self, tDs=None):
        """Create full nparray corresponding to the symmetric tensor."""
        return self.config.backend.to_numpy(self.to_dense(tDs=tDs))

    def to_number(self):
        """
        Return first number in the first (unsorted) block.
        Mainly used for rank-0 tensor with 1 block of size 1.

        Return 0 if there are no blocks.
        """
        return self.config.backend.first_element(self.A)

    def norm(self, ord='fro'):
        """
        Norm of the rensor.

        Parameters
        ----------
        ord: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        return self.config.backend.norm(self.A, ord=ord)

    def norm_diff(self, other, ord='fro'):
        """
        Norm of the difference of the two tensors.

        Parameters
        ----------
        other: Tensor

        ord: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        if not np.all(self.s == other.s):
            logger.exception('Signs do not match')
            raise FatalError
        return self.config.backend.norm_diff(self.A, other.A, ord)

    def entropy(self, axes=(0, 1), alpha=1):
        r"""
        Calculate entropy from tensor.

        If diagonal, calculates entropy treating S^2 as probabilities. Normalizes S^2 if neccesary.
        If not diagonal, calculates svd first to get the diagonal S.
        Use base-2 log.

        Parameters
        ----------
        axes: tuple
            how to split the tensor for svd

        alpha: float
            Order of Renyi entropy.
            alpha=1 is von Neuman: Entropy -Tr(S^2 log2(S^2))
            otherwise: 1/(1-alpha) log2(Tr(S^(2*alpha)))

        Returns
        -------
        entropy, minimal singular value, normalization : float64
        """
        out_l, out_r = _clean_axes(axes)
        self._test_axes_split(out_l, out_r)
        
        if not self.isdiag:
            Am, *_ = self._merge_to_matrix(out_l, out_r, news_l=-1, news_r=1)
            Sm = self.config.backend.svd_S(Am)
        else:
            Sm = {t: self.config.backend.diag_get(x) for t, x in self.A.items()}
        entropy, Smin, normalization = self.config.backend.entropy(Sm, alpha=alpha)
        return entropy, Smin, normalization

    def max_abs(self):
        """
        Largest element by magnitude.

        Returns
        -------
        max_abs : float64
        """
        return self.config.backend.max_abs(self.A)    

    #############################
    #     linear operations     #
    #############################

    def __mul__(self, other):
        """
        Multiply tensor by a number, use: number * tensor.

        Parameters
        ----------
        other: number

        Returns
        -------
        tensor : Tensor
            result of multipcilation as a new tensor
        """
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A= {ind: other * x for ind, x in self.A.items()}
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    def __rmul__(self, other):
        """
        Multiply tensor by a number, use: tensor * number.

        Parameters
        ----------
        other: number

        Returns
        -------
        tensor : Tensor
            result of multipcilation as a new tensor
        """
        return self.__mul__(other)

    def __pow__(self, exponent):
        """
        Element-wise exponent of tensor, use: tensor ** exponent.

        Parameters
        ----------
        exponent: number

        Returns
        -------
        tensor : Tensor
            result of element-wise exponentiation as a new tensor
        """
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A= {ind: x**exponent for ind, x in self.A.items()}
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    def __truediv__(self, other):
        """
        Divide tensor by a scalar, use: tensor / scalar.

        Parameters
        ----------
        other: scalar

        Returns
        -------
        tensor : Tensor
            result of element-wise division  as a new tensor
        """
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A= {ind: x / other for ind, x in self.A.items()}
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()  
        return a

    def __add__(self, other):
        """
        Add two tensors, use: tensor + tensor.

        Signatures and total charges should match.

        Parameters
        ----------
        other: Tensor

        Returns
        -------
        tensor : Tensor
            result of addition as a new tensor
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            logger.exception('Tensor signatures do not match')
            raise FatalError
        meta = _common_keys(self.A, other.A)
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = a.config.backend.add(self.A, other.A, meta)
        a._calculate_tDset()
        return a

    def __sub__(self, other):
        """
        Subtract two tensors, use: tensor - tensor.

        Both signatures and total charges should match.

        Parameters
        ----------
        other: Tensor

        Returns
        -------
        tensor : Tensor
            result of subtraction as a new tensor
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            logger.exception('Tensors do not match')
            raise FatalError
        meta = _common_keys(self.A, other.A)
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = a.config.backend.sub(self.A, other.A, meta)
        a._calculate_tDset()
        return a

    def apxb(self, other, x=1):
        """
        Directly calculate tensor + x * other tensor

        Signatures and total charges should match.

        Parameters
        ----------
        other: Tensor
        x : number

        Returns
        -------
        tensor : Tensor
            result of addition as a new tensor
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            logger.exception('Tensors do not match')
            raise FatalError
        meta = _common_keys(self.A, other.A)
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = a.config.backend.apxb(self.A, other.A, x, meta)
        a._calculate_tDset()
        return a

    #############################
    #     tensor operations     #
    #############################

    def transpose(self, axes=(1,0)):
        r"""
        Return transposed tensor.

        Parameters
        ----------
        axes: tuple
            New order of the legs.

        Returns
        -------
        tensor : Tensor
        """
        if not self.isdiag:
            order = np.array(axes, dtype=np.intp)
            a = Tensor(config=self.config, s=self.s[order], n=self.n, isdiag=self.isdiag)
            a.tset = self.tset[:, order, :]
            a.Dset = self.Dset[:, order]
            meta_transpose = tuple((tuple(old.flat), tuple(new.flat)) for old, new in zip(self.tset, a.tset))
            a.A = a.config.backend.transpose(self.A, axes, meta_transpose)
            return a
        else:
            return self.copy()
    
    def moveaxis(self, source, destination, inplace=False):
        r"""
        Change the position of a single axis of the tensor. 
        
        Operation can be done in-place.

        Parameters
        ----------
        source, destination: ints
        """
        a = self if inplace else self.copy()
        source = source + a.ndim if source < 0 else source
        destination = destination + a.ndim if destination < 0 else destination
        if source == destination:
            return a
        axes = list(range(a.ndim))
        axes.insert(destination, axes.pop(source))
        axes = np.array(axes, dtype=np.intp)
        new_tset = self.tset[:, axes, :]
        meta_moveaxis = tuple((tuple(old.flat), tuple(new.flat)) for old, new in zip(self.tset, new_tset))
        a.A = self.config.backend.moveaxis_local(a.A, source, destination, meta_moveaxis)
        a.tset = new_tset
        a.Dset = a.Dset[:, axes]
        a.s = a.s[axes]
        return a

    def diag(self, s0=1):
        """
        Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa.

        Parameters
        ----------
            s0: +1 or -1
                while transforming diagonal tensor into 2d tensor, one has to select signature (s0, -s0)
        """
        if self.isdiag:
            a = Tensor(config=self.conf, s=(s0, -s0), n=self.n, isdiag=False)
            for ind in self.A:
                nind = ind + ind
                a.set_block(ts=nind, val=self.conf.backend.diag_create(self.A[ind]))
            return a
        elif self.ndim == 2 and sum(np.abs(self.n)) == 0 and sum(self.s) == 0:
            a = Tensor(config=self.conf, isdiag=True)
            for ind in self.tset:
                if np.all(ind[0, :] == ind[1, :]):
                    nind = tuple(ind[0, :].flat)
                    a.set_block(ts=nind, val=self.conf.backend.diag_get(self.A[tuple(ind.flat)]))
            return a
        else:
            logger.exception('Tensor cannot be changed into a diagonal one')

    def conj(self):
        """
        Return conjugated tensor.

        Changes sign of signature s and total charge n, as well as complex conjugate each block.

        Returns
        -------
        tensor : Tensor
        """
        a = Tensor(config=self.conf, s=-self.s, n=-self.n, isdiag=self.isdiag)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        a.A = a.config.backend.conj(self.A)
        return a

    def swap_gate(self, axes, fermionic=()):
        """
        Return tensor after application of the swap gate.

        Multiply the block with odd charges on swaped legs by -1.
        If one of the axes is -1, then swap with charge n.

        TEST IT

        Parameters
        ----------
        axes: tuple
            two legs to be swaped

        fermionic: tuple
            which symmetries are fermionic

        Returns
        -------
        tensor : Tensor
        """
        if any(fermionic):
            fermionic = np.array(fermionic, dtype=bool)
            axes = sorted(list(axes))
            a = self.copy()
            if axes[0] == axes[1]:
                logger.exception('Cannot sweep the same index')
                raise FatalError
            if not self.isdiag:
                if (axes[0] == -1) and (np.sum(a.n[fermionic]) % 2 == 1):  # swap gate with local a.n
                    for ind in a.tset:
                        if np.sum(ind[axes[1], fermionic]) % 2 == 1:
                            ind = tuple(ind.flat)
                            a.A[ind] = -a.A[ind]
                else:  # axes[0] != axes[1]:  # swap gate on 2 legs
                    for ind in a.tset:
                        if (np.sum(ind[axes[0], fermionic]) % 2 == 1) and (np.sum(ind[axes[1], fermionic]) % 2 == 1):
                            a.A[ind] = -a.A[ind]
            else:
                for ind in a.tset:
                    if (np.sum(ind[axes[0], fermionic]) % 2 == 1):
                        a.A[ind] = -a.A[ind]
            return a
        else:
            return self

    def invsqrt(self):
        """ Element-wise 1/sqrt(A)"""
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        if not a.isdiag:
            a.A = self.config.backend.invsqrt(self.A)
        else:
            a.A = self.config.backend.invsqrt_diag(self.A)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    def inv(self):
        """ Element-wise 1/sqrt(A)"""
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        if not a.isdiag:
            a.A = self.config.backend.inv(self.A)
        else:
            a.A = self.config.backend.inv_diag(self.A)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    def exp(self, step=1.):
        """ Element-wise exp(step * A)"""
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        if not a.isdiag:
            a.A = self.config.backend.exp(self.A, step)
        else:
            a.A = self.config.backend.exp_diag(self.A, step)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    def sqrt(self):
        """ Element-wise sqrt"""
        a = self.empty(s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = self.config.backend.sqrt(self.A)
        a.tset = self.tset.copy()
        a.Dset = self.Dset.copy()
        return a

    ##################################
    #     contraction operations     #
    ##################################

    def scalar(self, other):
        r""" Compute scalar product x=(a, b) of two tensors.

            Note that the first one is conjugated.

            Parameters
            ----------
            other: Tensor

            Returns
            -------
            x: number
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            logger.exception('Tensors do not match')
            raise FatalError
        k12, _, _ = _common_keys(self.A, other.A)
        return self.config.backend.scalar(self.A, other.A, k12)

    def trace(self, axes=(0, 1)):
        """
        Compute trace of legs specified by axes.

        Parameters
        ----------
            axes: tuple
            Legs to be traced out, e.g axes=(0, 1); or axes=((2, 3, 4), (0, 1, 5))
        
        Returns
        -------
            tansor: Tensor
        """
        in1, in2 = _clean_axes(axes)  # contracted legs

        if len(in1) != len(in2):
            logger.exception('Number of axis to trace should be the same')
            raise FatalError
        if len(in1) == 0:
            return self

        in12 = in1 + in2
        out = tuple(ii for ii in range(self.ndim) if ii not in in12)
        order = in12 + out
        ain1 = np.array(in1, dtype=np.intp)
        ain2 = np.array(in2, dtype=np.intp)
        aout = np.array(out, dtype=np.intp)

        if not all(self.s[ain1] == -self.s[ain2]):
            logger.exception('Signs do not match')
            raise FatalError

        lt = len(self.tset)
        t1 = self.tset[:, ain1, :].reshape(lt, -1)
        t2 = self.tset[:, ain2, :].reshape(lt, -1)
        to = self.tset[:, aout, :].reshape(lt, -1)
        D1 = self.Dset[:, ain1] 
        D2 = self.Dset[:, ain2] 
        D3 = self.Dset[:, aout] 
        pD1 = np.prod(D1, axis=1).reshape(lt, 1)
        pD2 = np.prod(D2, axis=1).reshape(lt, 1)
        ind = (np.all(t1==t2, axis=1)).nonzero()[0]
        Drsh = np.hstack([pD1, pD2, D3])

        if not np.all(D1[ind] == D2[ind]):
            logger.exception('Not all bond dimensions of the traced legs match')
            raise FatalError

        meta = [(tuple(to[n]), tuple(self.tset[n].flat), tuple(Drsh[n])) for n in ind]
        a = Tensor(config=self.config, s=self.s[aout], n=self.n)
        a.A = a.config.backend.trace(self.A, order, meta)
        a._calculate_tDset()
        return a

    def dot(self, other, axes, conj=(0, 0)):
        r""" Compute dot product of two tensor along specified axes.

            Outgoing legs ordered such that first come remaining legs of the first tensor in the original order,
            and than those of the second tensor.

            Parameters
            ----------
            other: Tensor

            axes: tuple
                legs of both tensors (for each it is specified by int or tuple of ints)
                e.g. axes=(0, 3), axes=((0, 3), (1, 2))

            conj: tuple
                shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1).
        """
        a_con, b_con = _clean_axes(axes)  # contracted legs
        a_out = tuple(ii for ii in range(self.ndim) if ii not in a_con)  # outgoing legs
        b_out = tuple(ii for ii in range(other.ndim) if ii not in b_con)
        
        na_con, na_out = np.array(a_con, dtype=np.intp), np.array(a_out, dtype=np.intp)
        nb_con, nb_out = np.array(b_con, dtype=np.intp), np.array(b_out, dtype=np.intp)

        conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])
        if not all(self.s[na_con] == -other.s[nb_con] * conja * conjb):
            if self.isdiag:  # if diagonal tensor, than freely changes the signature by by a factor of -1
                self.s *= -1
            elif other.isdiag:
                other.s *= -1
            else:
                logger.exception('Signs do not match')
                raise FatalError

        c_n = np.vstack([self.n, other.n]).reshape(1,2,-1)
        c_s = np.array([conja, conjb], dtype=int)
        c_n = self.config.sym.fuse(c_n, c_s, 1)

        t_a_con, t_b_con = self.tset[:, na_con, :], other.tset[:, nb_con, :]
        inda, indb = _indices_common_rows(t_a_con, t_b_con)
       
        Am, ls_l, _, ua_l, ua_r = self._merge_to_matrix(a_out, a_con, conja, -conja, inda, sort_r=True)
        Bm, _, ls_r, ub_l, ub_r = other._merge_to_matrix(b_con, b_out, conjb, -conjb, indb)

        meta_dot = tuple((al + br, al + ar, bl + br)  for al, ar, bl, br in zip(ua_l, ua_r, ub_l, ub_r))

        if not ua_r == ub_l:
            logger.exception('Something went wrong in matching the indices of the two tensors')
            raise FatalError

        Cm = self.config.backend.dot(Am, Bm, conj, meta_dot)
        c_s = np.hstack([conja * self.s[na_out], conjb * other.s[nb_out]])
        c = Tensor(config=self.config, s=c_s, n=c_n)
        c.A = self._unmerge_from_matrix(Cm, ls_l, ls_r)
        c._calculate_tDset()
        return c

    ###########################
    #     spliting tensor     #
    ###########################

    def split_svd(self, axes, sU=1, nU=True, Uaxis=-1, Vaxis=0, tol=0, D_block=np.inf, \
        D_total=np.inf, truncated_svd=False, truncated_nbit=60, truncated_kfac=6):
        r"""
        Split tensor into U @ S @ V using svd. Can truncate smallest singular values.

        Truncate based on relative tolerance, bond dimension of each block,
        and total bond dimension from all blocks (whichever gives smaller bond dimension).
        By default, do not truncate. 
        Charge of tensor is attached to U if nU and to V if not nU

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as 
            their final order.

        sU: int
            signature of the new leg in U; equal 1 or -1. Default is 1.

        Uaxis, Vaxis: int
            specify which leg of U and V tensors are connecting with S. By default 
            it is the last leg of U and the first of V.

        tol: float
            relative tolerance of singular values below which to truncate.

        D_block: int
            largest number of singular values to keep in a single block.

        D_total: int
            largest total number of singular values to keep.

        truncated_svd: bool
            flag to employ truncated-svd algorithm.

        truncated_nbit, truncated_kfac: int
            parameters of the truncated-svd algorithm.

        Returns
        -------
        U, S, V: Tensor
            U and V are unitary projectors. S is diagonal.
        """
        out_l, out_r = _clean_axes(axes)
        self._test_axes_split(out_l, out_r)

        Am, ls_l, ls_r, ul, ur = self._merge_to_matrix(out_l, out_r, news_l=-sU, news_r=sU)
        
        if nU:
            meta = tuple((il+ir, il+ir, ir, ir+ir) for il, ir in zip(ul, ur))
            n_l, n_r = self.n, 0*self.n
        else:
            meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
            n_l, n_r = 0*self.n, self.n

        opts = {'truncated_svd': truncated_svd, 'D_block': D_block,
               'nbit': truncated_nbit, 'kfac':truncated_kfac,
               'tol': tol, 'D_total': D_total}

        Um, Sm, Vm = self.config.backend.svd(Am, meta, opts)

        U = Tensor(config=self.config, s=ls_l.s + (sU,), n=n_l)
        S = Tensor(config=self.config, s=(-sU, sU), isdiag=True)
        V = Tensor(config=self.config, s=(-sU,) + ls_r.s, n=n_r)

        ls_s = _Leg_struct(self.config)
        ls_s.leg_struct_for_truncation(Sm, opts, 'svd')

        U.A = self._unmerge_from_matrix(Um, ls_l, ls_s)
        S.A = self._unmerge_from_diagonal(Sm, ls_s)
        V.A = self._unmerge_from_matrix(Vm, ls_s, ls_r)

        U._calculate_tDset()
        S._calculate_tDset()
        V._calculate_tDset()
        U.moveaxis(source=-1, destination=Uaxis, inplace=True)
        V.moveaxis(source=0, destination=Vaxis, inplace=True)
        return U, S, V

    def split_qr(self, axes, sQ=1, Qaxis=-1, Raxis=0):
        r"""
        Split tensor using qr decomposition, tensor = Q * R.

        Charge of R is zero.

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as their final order.

        sQ: int
            signature of connecting leg in Q; equal 1 or -1. Default is 1.

        Qaxis, Raxis: int
            specify which leg of Q and R tensors are connecting to the other tensor. By delault it is the last leg of Q and the first of R.

        Returns
        -------
            Q, R: Tensor
        """
        out_l, out_r = _clean_axes(axes)
        self._test_axes_split(out_l, out_r)

        Am, ls_l, ls_r, ul, ur = self._merge_to_matrix(out_l, out_r, news_l=-sQ, news_r=sQ)
        
        meta = tuple((l+r, l+r, r+r) for l, r in zip(ul, ur))
        Qm, Rm = self.config.backend.qr(Am, meta)
        
        Qs = tuple(self.s[lg] for lg in out_l) + (sQ,)
        Rs = (-sQ,) + tuple(self.s[lg] for lg in out_r)
        Q = Tensor(config=self.config, s=Qs, n=self.n)
        R = Tensor(config=self.config, s=Rs)

        ls = _Leg_struct(self.config, -sQ, -sQ)
        ls.leg_struct_trivial(Rm, 0)

        Q.A = self._unmerge_from_matrix(Qm, ls_l, ls)
        R.A = self._unmerge_from_matrix(Rm, ls, ls_r)

        Q._calculate_tDset()
        R._calculate_tDset()

        Q.moveaxis(source=-1, destination=Qaxis, inplace=True)
        R.moveaxis(source=0, destination=Raxis, inplace=True)
        return Q, R

    def split_eigh(self, axes=(0, 1), sU=1, Uaxis=-1, tol=0, D_block=np.inf, D_total=np.inf):
        r"""
        Split tensor using eig, tensor = U * S * U^dag. Truncate smallest eigenvalues if neccesary.

        Tensor should be hermitian and has charge 0.
        Truncate using (whichever gives smaller bond dimension) relative tolerance, bond dimension of each block, and total bond dimension from all blocks.
        By default do not truncate. Truncate based on tolerance only if some eigenvalue is positive -- than all negative ones are discarded.
        Function primarly intended to be used for positively defined tensors.

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as their final order.

        sU: int
            signature of connecting leg in U equall 1 or -1. Default is 1.

        Uaxis: int
            specify which leg of U is the new connecting leg. By delault it is the last leg.

        tol: float
            relative tolerance of singular values below which to truncate.

        D_block: int
            largest number of singular values to keep in a single block.

        D_total: int
            largest total number of singular values to keep.

        Returns
        -------
            S, U: Tensor
                U is unitary projector. S is diagonal.
        """
        out_l, out_r = _clean_axes(axes)
        self._test_axes_split(out_l, out_r)

        if np.any(self.n != 0):
            logger.exception('Charge should be zero')
            raise FatalError

        Am, ls_l, ls_r, ul, ur = self._merge_to_matrix(out_l, out_r, news_l=-sU, news_r=sU)
        
        if not ul == ur:
            logger.exception('Something went wrong in matching the indices of the two tensors')
            raise FatalError

        # meta = (indA, indS, indU)
        meta = tuple((l+r, l, l+r) for l, r in zip(ul, ur))
        Sm, Um = self.config.backend.eigh(Am, meta)
        
        opts = {'D_block': D_block, 'tol': tol, 'D_total': D_total}
        ls_s = _Leg_struct(self.config, -sU, -sU)
        ls_s.leg_struct_for_truncation(Sm, opts, 'eigh')
                
        Us = tuple(self.s[lg] for lg in out_l) + (sU,)
        
        S = Tensor(config=self.config, s=(-sU, sU), isdiag=True)
        U = Tensor(config=self.config, s=Us)

        U.A = self._unmerge_from_matrix(Um, ls_l, ls_s)
        S.A = self._unmerge_from_diagonal(Sm, ls_s)

        U._calculate_tDset()
        S._calculate_tDset()

        U.moveaxis(source=-1, destination=Uaxis, inplace=True)
        return S, U

    ##############################
    #     merging operations     #
    ##############################
    
    def _merge_to_matrix(self, out_l, out_r, news_l, news_r, ind=slice(None), sort_r=False):
        order = out_l + out_r
        legs_l, legs_r = np.array(out_l, np.int), np.array(out_r, np.int)
        tset, Dset = self.tset[ind], self.Dset[ind] 
        t_l, t_r = tset[:, legs_l, :], tset[:, legs_r, :]
        D_l, D_r = Dset[:, legs_l], Dset[:, legs_r]
        s_l, s_r = self.s[legs_l], self.s[legs_r]
        Deff_l, Deff_r = np.prod(D_l, axis=1), np.prod(D_r, axis=1)
        
        teff_l = self.config.sym.fuse(t_l, s_l, news_l)
        teff_r = self.config.sym.fuse(t_r, s_r, news_r)
        t_new = np.hstack([teff_l, teff_r])

        ls_l = _Leg_struct(self.config, s_l, news_l)
        ls_r = _Leg_struct(self.config, s_r, news_r)
        ls_l.leg_struct_for_merged(teff_l, t_l, Deff_l, D_l)
        ls_r.leg_struct_for_merged(teff_r, t_r, Deff_r, D_r)

        u_new, iu_new = np.unique(t_new, return_index=True, axis=0)
        u_new_l, u_new_r = teff_l[iu_new], teff_r[iu_new]

        if sort_r and len(u_new_r) > 1:
            iu_r = np.lexsort(u_new_r.T[::-1])
            u_new, u_new_l, u_new_r = u_new[iu_r], u_new_l[iu_r], u_new_r[iu_r]
        
        u_new_l = tuple(tuple(x.flat) for x in u_new_l)
        u_new_r = tuple(tuple(x.flat) for x in u_new_r)
        # meta_new = ((unew, Dnew), ...)
        meta_new = tuple((tuple(u.flat), (ls_l.D[l], ls_r.D[r])) for u, l, r in zip(u_new, u_new_l, u_new_r))
        # meta_mrg = ((tnew, told, Dslc_l, D_l, Dslc_r, D_r), ...)
        meta_mrg = tuple((tuple(tn.flat), tuple(to.flat), *ls_l.dec[tuple(tel.flat)][tuple(tl.flat)][:2], *ls_r.dec[tuple(ter.flat)][tuple(tr.flat)][:2]) 
            for tn, to, tel, tl, ter, tr in zip(t_new, tset, teff_l, t_l, teff_r, t_r))
       
        Anew = self.config.backend.merge_to_matrix(self.A, order, meta_new, meta_mrg, self.config.dtype)
        return Anew, ls_l, ls_r, u_new_l, u_new_r

    def _unmerge_from_matrix(self, A, ls_l, ls_r):
        meta = []
        for il, ir in itertools.product(ls_l.dec, ls_r.dec):
            ic = il + ir
            if ic in A:
                for (tl, (sl, _, Dl)), (tr, (sr, _, Dr)) in itertools.product(ls_l.dec[il].items(), ls_r.dec[ir].items()):
                    meta.append((tl + tr, ic, sl, sr, Dl+Dr))
        return self.config.backend.unmerge_from_matrix(A, meta)

    def _unmerge_from_diagonal(self, A, ls):
        meta = tuple((ta + ta, ia, sa) for ia in ls.dec for ta, (sa, _, _) in ls.dec[ia].items())
        Anew = self.config.backend.unmerge_from_diagonal(A, meta)
        return {ind: self.config.backend.diag_create(Anew[ind]) for ind in Anew}
        
    ##########################
    #    fusing operations   #
    ##########################

    def group_legs(self, axes, new_s=None):
        """
        Permutes tensor legs. Next, fuse a specified group of legs into a new single leg.
    
        If no internal tuple if given, fuse the given indices. 
        New leg is placed at the position of the first fused oned.

        Parameters
        ----------
        axes: tuple 
            tuple of leg indices for transpose. Group of legs to be fused forms inner tuple.

        new_s: int
            signature of a new leg. If not given, the signature of the first fused leg is given.

        Returns
        -------
        tensor : Tensor

        Example
        -------
        For tensor with 5 legs: tensor.fuse_legs1(axes=(2, 0, (1, 4), 3))
        tensor.fuse_legs1(axes=(2, 0)) is equivalent to tensor.fuse_legs1(axes=(1, (2, 0), 3, 4))
        """
        if self.isdiag:
            logger.exception('Cannot group legs of a diagonal tensor')
            raise FatalError

        ituple = [ii for ii, ax in enumerate(axes) if isinstance(ax, tuple)]
        if len(ituple) == 1:
            ig = ituple[0]
            al, ag, ar = axes[:ig], axes[ig], axes[ig+1:]
        elif len(ituple) == 0: 
            al = tuple(ii for ii in range(axes[0]) if ii not in axes)
            ar = tuple(ii for ii in range(axes[0]+1, self.ndim) if ii not in axes)
            ag = axes
            ig = len(al)
        else:
            logger.exception('Too many groups to fuse')
            raise FatalError
        if len(ag) < 2:
            logger.exception('Need at least two legs to fuse')
            raise FatalError

        order = al+ag+ar  # order for permute
        legs_l = np.array(al, dtype=np.intp)
        legs_r = np.array(ar, dtype=np.intp)
        legs_c = np.array(ag, dtype=np.intp)

        if new_s is None:
            new_s = self.s[ag[0]]
        
        new_ndim = len(al) + 1 + len(ar)

        t_grp = self.tset[:, legs_c, :]
        D_grp = self.Dset[:, legs_c]
        s_grp = self.s[legs_c]
        t_eff = self.config.sym.fuse(t_grp, s_grp, new_s)
        D_eff = np.prod(D_grp, axis=1)

        D_rsh = np.empty((len(self.A), new_ndim), dtype=int)
        D_rsh[:, :ig] = self.Dset[:, legs_l]
        D_rsh[:, ig] = D_eff
        D_rsh[:, ig+1:] = self.Dset[:, legs_r]

        ls_c = _Leg_struct(self.config, s_grp, new_s)
        ls_c.leg_struct_for_merged(t_eff, t_grp, D_eff, D_grp)

        t_new = np.empty((len(self.A), new_ndim, self.config.sym.nsym), dtype=int)
        t_new[:, :ig, :] = self.tset[:, legs_l, :]
        t_new[:, ig, :] = t_eff
        t_new[:, ig+1:, :] = self.tset[:, legs_r, :]

        t_new = t_new.reshape(len(t_new), -1)
        u_new, iu_new = np.unique(t_new, return_index=True, axis=0)
        Du_new = D_rsh[iu_new]
        Du_new[:, ig] = np.array([ls_c.D[tuple(t_eff[ii].flat)] for ii in iu_new], dtype=int)
        
        # meta_new = ((u, Du), ...)
        meta_new = tuple((tuple(u.flat), tuple(Du)) for u, Du in zip(u_new, Du_new))
        # meta_mrg = ((tn, Ds, to, Do), ...)
        meta_mrg = tuple((tuple(tn.flat), ls_c.dec[tuple(te.flat)][tuple(tg.flat)][0], tuple(to.flat), tuple(Do)) 
            for tn, te, tg, to, Do in zip(t_new, t_eff, t_grp, self.tset, D_rsh))

        c = self.empty(s=tuple(self.s[legs_l]) + (new_s,) + tuple(self.s[legs_r]), n=self.n, isdiag=self.isdiag)
        c.A = self.config.backend.merge_one_leg(self.A, ig, order, meta_new , meta_mrg, self.config.dtype) 
        c._calculate_tDset()
        c.lss[ig] = ls_c  
        for nnew, nold in enumerate(al+ (-1,) + ar):
            if nold in self.lss:
                c.lss[nnew] = self.lss[nold].copy()
        return c

    def ungroup_leg(self, axis):
        """
        Unfuse a single tensor leg.

        New legs are inserted in place of the unfused one.

        Parameters
        ----------
        axis: int
            index of leg to ungroup.

        Returns
        -------
        tensor : Tensor
        """
        try:
            ls = self.lss[axis]
        except KeyError:
            return self

        meta = []
        for tt, DD in zip(self.tset, self.Dset):
            tl = tuple(tt[:axis, :].flat)
            tc = tuple(tt[axis, :].flat)
            tr = tuple(tt[axis+1:, :].flat)
            told = tuple(tt.flat)
            Dl = tuple(DD[:axis])
            Dr = tuple(DD[axis+1:])
            for tcom, (Dsl, _, Dc) in ls.dec[tc].items():
                tnew = tl + tcom + tr
                Dnew = Dl + Dc + Dr
                meta.append((told, tnew, Dsl, Dnew))
        meta = tuple(meta)
        s = tuple(self.s[:axis]) + ls.s + tuple(self.s[axis+1:])

        c = self.empty(s=s, n=self.n, isdiag=self.isdiag)
        c.A = self.config.backend.unmerge_one_leg(self.A, axis, self.ndim, meta)
        c._calculate_tDset()
        for ii in range(axis):
            if ii in self.lss:
                c.lss[ii]=self.lss[ii].copy()
        for ii in range(axis+1, self.ndim):
            if ii in self.lss:
                c.lss[ii+ls.ndim]=self.lss[ii].copy()
        return c

    #################
    #     tests     #
    #################

    def is_independent(self, other):
        """
        Test if all elements of two tensors are independent objects in memory.
        """
        test = []
        test.append(self is other)
        test.append(self.A is other.A)
        test.append(self.n is other.n)
        test.append(self.s is other.s)
        test.append(self.tset is other.tset)
        test.append(self.Dset is other.Dset)
        for key in self.A.keys():
            if key in other.A:
                test.append(self.config.backend.is_independent(self.A[key], other.A[key]))
        return not any(test)

    def is_consistent(self):
        """
        Test if tset corresponds to indices of A; and is f(s * t - n) == 0 is satisfied for all blocks.
        """
        test = []
        for ind, D in zip(self.tset, self.Dset):
            ind = tuple(ind.flat)
            test.append(ind in self.A)
            test.append(self.config.backend.get_shape(self.A[ind]) == tuple(D))
        test.append(len(self.tset) == len(self.A))
        test.append(len(self.tset) == len(self.Dset))

        test.append(np.all(self.config.sym.fuse(self.tset, self.s, 1) == self.n))
        for n in range(self.ndim):
            self.get_leg_tD(n)

        return all(test)

    ########################
    #     aux function     #
    ########################

    def _test_axes_split(self, out_l, out_r):
        if not (self.ndim == len(out_l) + len(out_r)):
            logger.exception('Two few indices in axes')
            raise FatalError
        if not (sorted(set(out_l+out_r)) == list(range(self.ndim))):
            logger.exception('Repeated axis')
            raise FatalError

    def _calculate_tDset(self):
        self.tset = np.array([ind for ind in self.A], dtype=int).reshape(len(self.A), self.ndim, self.config.sym.nsym)
        self.Dset = np.array([self.config.backend.get_shape(self.A[ind]) for ind in self.A], dtype=int).reshape(len(self.A), self.ndim)


class _Leg_struct:
    r"""
    Information about internal structure of leg resulting from fusions.
    """ 
    def __init__(self, config=None, s=(), news=1):
        try:
            self.ndim = len(s)  # number of fused legs
            self.s = tuple(s)  # signature of fused legs
        except TypeError:
            self.s = (s,)
            self.ndim = 1
        self.config = config
        self.news = news # signature of effective leg
        self.D = {}
        self.dec = {}  # leg's structure/ decomposition

    def copy(self):
        ls = _Leg_struct(s=self.s, news=self.news)
        for te, de in self.dec.items():
            ls.dec[te] = {to: Do for to, Do in de.items()}
        ls.D = {t: D for t, D in self.D.items()}
        return ls

    def show(self):
        print("Leg structure: fused = ", self.nlegs)
        for te, de in self.dec.items():
            print(te, ":")
            for to, Do in de.items():
                print("   ",to, ":", Do)
    
    def leg_struct_for_merged(self, teff, tlegs, Deff, Dlegs):
        """ Calculate meta-information about bond dimensions for merging into one leg. """
        shape_t =list(tlegs.shape)
        shape_t[1] = shape_t[1] + 1
        tcom = np.empty(shape_t, dtype=int)
        tcom[:, 0, :] = teff
        tcom[:, 1:, :] = tlegs
        tcom = tcom.reshape((shape_t[0], shape_t[1]*shape_t[2])) 
        ucom, icom = np.unique(tcom, return_index=True, axis=0)
        Dlow = 0
        for ii, tt in zip(icom, ucom):
            t0 = tuple(tt[:self.config.sym.nsym])
            t1 = tuple(tt[self.config.sym.nsym:])
            if t0 not in self.dec:
                self.dec[t0] = {}
                Dlow = 0
            Dtop = Dlow + Deff[ii]
            self.dec[t0][t1] = ((Dlow, Dtop), Deff[ii], tuple(Dlegs[ii]))
            Dlow = Dtop
            self.D[t0] = Dtop

    def leg_struct_trivial(self, A, axis):
        """ Meta-information for single leg. """
        nsym = self.config.sym.nsym
        for ind, val in A.items():
            t = ind[nsym * axis: nsym*(axis + 1)]
            D = self.config.backend.get_shape(val)[axis]
            self.dec[t] = {t: ((0, D), D, (D,))}                        

    def leg_struct_for_truncation(self, A, opts, sorting='svd'):
        r"""Gives slices for truncation of 1d matrices according to tolerance, D_block, D_total.

        A should be dict of ordered 1d arrays.
        Sorting gives information about ordering outputed by a particular splitting funcion:
        Usual convention is that for svd A[ind][0] is largest; and for eigh A[ind][-1] is largest.
        """
        maxS = self.config.backend.maximum(A)
        Dmax, D_keep = {}, {}
        for ind in A:
            Dmax[ind] = self.config.backend.get_size(A[ind])
            D_keep[ind] = min(opts['D_block'], Dmax[ind])
        if (opts['tol'] > 0) and (maxS > 0):  # truncate to relative tolerance
            for ind in D_keep:
                D_keep[ind] = min(D_keep[ind], self.config.backend.count_greater(A[ind], maxS * opts['tol']))
        if sum(D_keep[ind] for ind in D_keep) > opts['D_total']:  # truncate to total bond dimension
            order = self.config.backend.select_largest(A, D_keep, opts['D_total'], sorting)
            low = 0
            for ind in D_keep:
                high = low + D_keep[ind]
                D_keep[ind] = np.sum((low <= order) & (order < high))
                low = high
        for ind in D_keep:
            if D_keep[ind] > 0:
                Dslc = self.config.backend.range_largest(D_keep[ind], Dmax[ind], sorting)
                self.dec[ind] = {ind: (Dslc, D_keep[ind], (D_keep[ind],))}


def _clean_axes(axes):
    try:
        out_l = tuple(axes[0])
    except TypeError:
        out_l = (axes[0],)
    try:
        out_r = tuple(axes[1])
    except TypeError:
        out_r = (axes[1],)
    return out_l, out_r

def _common_keys(d1, d2):
    """
    Divide keys into: common, only in d1 and only in d2.

    Returns: keys12, keys1, keys2
    """
    s1 = set(d1)
    s2 = set(d2)
    return tuple(s1 & s2), tuple(s1 - s2), tuple(s2 - s1)

def _indices_common_rows(a, b):
    """
    Return indices of a that are in b, and indices of b that are in a.
    """
    la = [tuple(x.flat) for x in a]
    lb = [tuple(x.flat) for x in b]
    sa = set(la)
    sb = set(lb)
    ia = np.array([ii for ii, el in enumerate(la) if el in sb], dtype=np.intp)
    ib = np.array([ii for ii, el in enumerate(lb) if el in sa], dtype=np.intp)
    return ia, ib
