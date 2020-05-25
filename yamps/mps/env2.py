from yamps.tensor import ncon
from .mps import MpsError

################################################
#     environment for <bra|ket> operations     #
################################################


class Env2:
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
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.g = ket.g
        self.nr_phys = self.ket.nr_phys
        self.F = {}  # dict for environments

        if self.bra.nr_phys != self.ket.nr_phys:
            raise MpsError('bra and ket should have the same number of physical legs.')

        # set environments at boundaries
        ff = self.g.first
        ll = self.g.last
        tn = self.ket.A[ff]

        Ds = tn.match_legs(tensors=[self.bra.A[ff], self.ket.A[ff]],
                           legs=[self.bra.left[0], self.ket.left[0]],
                           conjs=[1, 0])
        self.F[(None, ff)] = tn.ones(**Ds)  # left boundary

        Ds = tn.match_legs(tensors=[self.ket.A[ll], self.bra.A[ll]],
                           legs=[self.ket.right[0], self.bra.right[0]],
                           conjs=[0, 1])
        self.F[(None, ll)] = tn.ones(**Ds)  # left boundary

    def update(self, n, towards):
        r"""
        Update environment including site n, in the direction given by towards.

        Parameters
        ----------
        n: int
            index of site to include to the environment
        towards : int
            towards which site (end) is the environment facing.
        """

        nnext, leg, nprev = self.g.from_site(n, towards)

        if self.nr_phys == 1 and leg == 1:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n]], ((2, 3, -1), (2, 1), (1, 3, -2)), (1, 0, 0))
        elif self.nr_phys == 1 and leg == 0:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.bra.A[n]], ((-1, 2, 1), (1, 3), (-2, 2, 3)), (0, 0, 1))
        elif self.nr_phys == 2 and leg == 1:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n]], ((2, 3, 4, -1), (2, 1), (1, 3, 4, -2)), (1, 0, 0))
        else:  # self.nr_phys == 2 and leg == 0:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.bra.A[n]], ((-1, 2, 3, 1), (1, 4), (-2, 2, 3, 4)), (0, 0, 1))

    def setup_to_last(self):
        r"""
        Setup all environments in the direction from first site to last site
        """
        for n in self.g.sweep(to='last'):
            self.update(n, towards=self.g.last)

    def setup_to_first(self):
        r"""
        Setup all environments in the direction from last site to first site
        """
        for n in self.g.sweep(to='first'):
            self.update(n, towards=self.g.first)

    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at nn bond

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.
            If None, it is measured at bond (outside, first site)

        Returns
        -------
        overlap : float or complex
        """
        if bd is None:
            bd = (None, self.g.first)
        return self.F[bd].dot(self.F[bd[::-1]], axes=((0, 1), (1, 0))).to_number()
