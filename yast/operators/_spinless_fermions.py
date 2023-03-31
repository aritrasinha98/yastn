from ..initialize import make_config
from ..sym import sym_Z2, sym_U1
from ..tensor import YastError, Tensor
from ._meta_operators import meta_operators

class SpinlessFermions(meta_operators):
    """ Predefine operators for spinless fermions. """

    def __init__(self, sym='U1', **kwargs):
        r""" 
        Standard operators for single fermionic species (2-dimensional Hilbert space).
        Defines identity, raising and lowering operators, and density operators.

        Other config parameters can be provided, see :meth:`yast.make_config` ,
        while :code:`fermionic` is always set to :code:`True` .

        Notes
        -----
        The following basis ordering and charge conventions are assumed
            
            * For both Z2 and U1, charge.
        """
        if sym not in ('Z2', 'U1'):
            raise YastError("For SpinlessFermions sym should be in ('Z2', 'U1').")
        kwargs['fermionic'] = True
        import_sym = {'Z2': sym_Z2, 'U1': sym_U1}
        kwargs['sym'] = import_sym[sym]
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')

    def I(self):
        """ Identity operator. """
        I = Tensor(config=self.config, s=self.s, n=0)
        I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return I

    def n(self):
        """ Particle number operator. """
        n = Tensor(config=self.config, s=self.s, n=0)
        n.set_block(ts=(0, 0), Ds=(1, 1), val=0)
        n.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return n

    def cp(self):
        """ Raising operator. """
        cp = Tensor(config=self.config, s=self.s, n=1)
        cp.set_block(ts=(1, 0), Ds=(1, 1), val=1)
        return cp

    def c(self):
        """ Lowering operator. """
        n = 1 if self._sym == 'Z2' else -1
        c = Tensor(config=self.config, s=self.s, n=n)
        c.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        return c

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'n': lambda j: self.n(),
                'c': lambda j: self.c(),
                'cp': lambda j: self.cp()}
