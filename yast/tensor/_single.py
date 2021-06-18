""" Linear operations and operations on a single yast tensor. """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _common_keys, _tarray, _Darray
from ._auxliary import _struct, _hard_fusion, _flip_sign_hard_fusion
from ._controls import YastError, _test_configs_match, _test_tensors_match

__all__ = ['conj', 'conj_blocks', 'flip_signature', 'transpose', 'moveaxis', 'diag', 'remove_zero_blocks',
           'absolute', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'apxb', 'add_leg',
           'copy', 'clone', 'detach', 'to', 'requires_grad_', 'fuse_legs', 'unfuse_legs']


def copy(a):
    """ Return a copy of the tensor.

        Warning: this might break autograd if you are using it.
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.copy(x) for ts, x in a.A.items()}
    return c


def clone(a):
    """ Return a copy of the tensor, tracking gradients. """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.clone(x) for ts, x in a.A.items()}
    return c


def detach(a, inplace=False):
    """ Detach tensor from autograd; Can be called inplace (?) """
    if inplace:
        for x in a.A.values():
            a.config.backend.detach_(x)
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.detach(x) for ts, x in a.A.items()}
    return c


def to(a, device):
    r"""
    Move the ``Tensor`` to ``device``. Returns a copy of the tensor on `device``.

    If the tensor already resides on ``device``, return a

    Parameters
    ----------
    device: str
        device identifier
    """
    if a.config.device == device:
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, device=device, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.move_to_device(a.A, device)
    return c


def requires_grad_(a, requires_grad=True):
    r"""
    Turn on recording of operations for the tensor for automatic differentiation.

    Parameters
    ----------
    requires_grad: bool
    """
    a.config.backend.requires_grad_(a.A, requires_grad=requires_grad)


def __mul__(a, number):
    """
    Multiply tensor by a number, use: number * tensor.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of multipcilation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: number * x for ind, x in a.A.items()}
    return c


def __rmul__(a, number):
    """
    Multiply tensor by a number, use: tensor * number.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of multipcilation as a new tensor
    """
    return __mul__(a, number)


def __pow__(a, exponent):
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x**exponent for ind, x in a.A.items()}
    return c


def __truediv__(a, number):
    """
    Divide tensor by a scalar, use: tensor / scalar.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of element-wise division  as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x / number for ind, x in a.A.items()}
    return c


def __add__(a, b):
    """
    Add two tensors, use: tensor + tensor.

    Signatures and total charges should match.

    Parameters
    ----------
    b: Tensor

    Returns
    -------
    tensor : Tensor
        result of addition as a new tensor
    """
    _test_configs_match(a, b)
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = c.config.backend.add(a.A, b.A, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        c.update_struct()
    return c


def __sub__(a, b):
    """
    Subtract two tensors, use: tensor - tensor.

    Both signatures and total charges should match.

    Parameters
    ----------
    b: Tensor

    Returns
    -------
    tensor : Tensor
        result of subtraction as a new tensor
    """
    _test_configs_match(a, b)
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = c.config.backend.sub(a.A, b.A, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        c.update_struct()
    return c


def apxb(a, b, x=1):
    """
    Directly calculate tensor: a + x * b

    Signatures and total charges should match.

    Parameters
    ----------
    a, b: yast tensors
    x : number

    Returns
    -------
    tensor : Tensor
        result of addition as a new tensor
    """
    _test_configs_match(a, b)
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = c.config.backend.apxb(a.A, b.A, x, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        c.update_struct()
    return c


def conj(a, inplace=False):
    """
    Return conjugated tensor.

    Changes sign of signature s and total charge n, as well as complex conjugate each block.

    Returns
    -------
    tensor : Tensor
    """
    an = np.array(a.struct.n, dtype=int).reshape(1, 1, -1)
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    new_hf = tuple(_flip_sign_hard_fusion(x) for x in a.hard_fusion)
    if inplace:
        c = a
        c.struct = struct
        a.hard_fusion = new_hf
    else:
        c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=new_hf, struct=struct)
    c.A = c.config.backend.conj(a.A, inplace)
    return c


def conj_blocks(a, inplace=False):
    """
    Conjugated each block, leaving symmetry structure unchanged.

    Returns
    -------
    tensor : Tensor
    """
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
                                        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = c.config.backend.conj(a.A, inplace)
    return c


def flip_signature(a, inplace=False):
    """
    Conjugated each block, leaving symmetry structure unchanged.

    Returns
    -------
    tensor : Tensor
    """
    an = np.array(a.struct.n, dtype=int).reshape(1, 1, -1)
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    new_hf = tuple(_flip_sign_hard_fusion(x) for x in a.hard_fusion)
    if inplace:
        a.struct = struct
        a.hard_fusion = new_hf
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=new_hf, struct=struct)
    c.A = {ind: a.config.backend.clone(a.A[ind]) for ind in a.A}
    return c


def transpose(a, axes, inplace=False):
    r"""
    Return transposed tensor.

    Operation can be done in-place, in which case copying of the data is not forced.
    Othersiwe, new tensor is created and the data are copied.

    Parameters
    ----------
    axes: tuple of ints
        New order of the legs. Should be a permutation of (0, 1, ..., ndim-1)
    """
    uaxes, = _unpack_axes(a, axes)
    order = np.array(uaxes, dtype=np.intp)
    new_meta_fusion = tuple(a.meta_fusion[ii] for ii in axes)
    new_hard_fusion = tuple(a.hard_fusion[ii] for ii in uaxes)
    news = tuple(a.struct.s[ii] for ii in uaxes)
    struct = _struct(s=news, n=a.struct.n)
    tset = _tarray(a)
    newt = tset[:, order, :]
    meta_transpose = tuple((told, tuple(tnew.flat)) for told, tnew in zip(a.struct.t, newt))
    if inplace:
        a.struct = struct
        a.meta_fusion = new_meta_fusion
        a.hard_fusion = new_hard_fusion
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=new_meta_fusion, hard_fusion=new_hard_fusion, struct=struct)
    c.A = c.config.backend.transpose(a.A, uaxes, meta_transpose, inplace)
    c.update_struct()
    return c


def moveaxis(a, source, destination, inplace=False):
    r"""
    Change the position of an axis (or a group of axes) of the tensor.

    Operation can be done in-place, in which case copying of the data is not forced.
    Othersiwe, new tensor is created and the data are copied. Calls transpose.

    Parameters
    ----------
    source, destination: ints
    """
    lsrc, ldst = _clear_axes(source, destination)
    lsrc = tuple(xx + a.mlegs if xx < 0 else xx for xx in lsrc)
    ldst = tuple(xx + a.mlegs if xx < 0 else xx for xx in ldst)
    if lsrc == ldst:
        return a if inplace else a.copy()
    axes = [ii for ii in range(a.mlegs) if ii not in lsrc]
    ds = sorted(((d, s) for d, s in zip(ldst, lsrc)))
    for d, s in ds:
        axes.insert(d, s)
    return transpose(a, axes, inplace)


def diag(a):
    """Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa. """
    if a.isdiag:
        c = a.__class__(config=a.config, isdiag=False, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_diag(a.A[ind]) for ind in a.A}
    elif a.nlegs == 2 and sum(abs(x) for x in a.struct.n) == 0 and sum(a.struct.s) == 0:
        c = a.__class__(config=a.config, isdiag=True, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_diag(a.A[ind]) for ind in a.A}
    else:
        raise YastError('Tensor cannot be changed into a diagonal one')
    # c.update_struct()
    return c


def absolute(a):
    """
    Return element-wise absolut value.

    Returns
    -------
    tansor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.absolute(a.A)
    return c


def real(a):
    """ return real part of tensor. Do not change dtype of yast.Tensor """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {t: a.config.backend.real(x) for t, x in a.A.items()}
    return c


def imag(a):
    """ return imaginary part of tensor """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {t: a.config.backend.imag(x) for t, x in a.A.items()}
    return c


def sqrt(a):
    """
    Return element-wise sqrt(A).

    Parameters
    ----------
    step: number

    Returns
    -------
    tansor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.sqrt(a.A)
    return c


def rsqrt(a, cutoff=0):
    """
    Return element-wise 1/sqrt(A).

    The tensor elements with absolut value below the cutoff are set to zero.

    Parameters
    ----------
        cutoff: float64
        Cut-off for (elementwise) pseudo-inverse.

    Returns
    -------
    tansor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.rsqrt_diag(a.A, cutoff=cutoff) if c.isdiag else a.config.backend.rsqrt(a.A, cutoff=cutoff)
    return c


def reciprocal(a, cutoff=0):
    """
    Return element-wise 1/A.

    The tensor elements with absolut value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: float64
        Cut-off for (elementwise) pseudo-inverse.

    Returns
    -------
    tansor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.reciprocal_diag(a.A, cutoff=cutoff) if c.isdiag else a.config.backend.reciprocal(a.A, cutoff=cutoff)
    return c


def exp(a, step=1.):
    """
    Return element-wise exp(step * A).

    This is calculated for existing blocks only.

    Parameters
    ----------
    step: number

    Returns
    -------
    tansor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.exp_diag(a.A, step) if c.isdiag else a.config.backend.exp(a.A, step)
    return c


def remove_zero_blocks(a, rtol=1e-12, atol=0, inplace=False):
    r"""
    Remove from the tensor blocks where all elements are below a cutoff.
    Cutoff is a combination of absolut tolerance and relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {k: t if inplace else a.config.backend.clone(t) for k, t in a.A.items() if a.config.backend.max_abs(t) > cutoff}
    c.update_struct()
    return c


def add_leg(a, axis=-1, s=1, t=None, inplace=False):
    r"""
    Creates a new auxiliary leg that explicitly carries charge (or part of it) associated with the tensor.

    Parameters
    ----------
        axis: int
            index of the new leg

        s : int
            signature of the new leg

        t : charge on the new leg.
            If None takes the charge of the tensor, making it zero

        inplace : bool
            If true, perform operation in place
    """
    if a.isdiag:
        raise YastError('Cannot add a new leg to a diagonal tensor.')
    tset, Dset = _tarray(a), _Darray(a)

    axis = axis % (a.mlegs + 1)

    new_meta_fusion = a.meta_fusion[:axis] + ((1,),) + a.meta_fusion[axis:]

    axis = sum(a.meta_fusion[ii][0] for ii in range(axis))  # unpack

    if s not in (-1, 1):
        raise YastError('The signature s should be equal to 1 or -1.')
    an = np.array(a.struct.n, dtype=int)
    if t is None:
        t = a.config.sym.fuse(an.reshape(1, 1, -1), np.array([1], dtype=int), -1)[0] if s == 1 else an  # s == -1
    else:
        t = a.config.sym.fuse(np.array(t, dtype=int).reshape(1, 1, -1), np.array([1], dtype=int), 1)[0]
    if len(t) != a.config.sym.NSYM:
        raise YastError('t does not have the proper number of symmetry charges')

    news = np.insert(np.array(a.struct.s, dtype=int), axis, s)
    newn = a.config.sym.fuse(np.hstack([an, t]).reshape(1, 2, -1), np.array([1, s], dtype=int), 1)[0]
    new_tset = np.insert(tset, axis, t, axis=1)
    new_Dset = np.insert(Dset, axis, 1, axis=1)

    news = tuple(news)
    newn = tuple(newn)
    new_tset = tuple(tuple(x.flat) for x in new_tset)
    new_Dset = tuple(tuple(x.flat) for x in new_Dset)

    c = a if inplace else a.clone()
    c.A = {tnew: a.config.backend.expand_dims(c.A[told], axis) for tnew, told in zip(new_tset, a.struct.t)}
    c.struct = _struct(new_tset, new_Dset, news, newn)
    c.meta_fusion = new_meta_fusion
    c.hard_fusion = c.hard_fusion[:axis] + (_hard_fusion(),) + c.hard_fusion[axis:]

    c.nlegs += 1
    c.mlegs += 1
    return c


def fuse_legs(a, axes, inplace=False):
    r"""
    Permutes tensor legs. Next, fuse groups of consecutive legs into new meta legs.

    Parameters
    ----------
    axes: tuple
        tuple of leg's indices for transpose. Groups of legs to be fused together form inner tuples.

    Returns
    -------
    tensor : Tensor

    Example
    -------
    tensor.fuse_legs(axes=(2, 0, (1, 4), 3)) gives 4 efective legs from original 5; with one metaly non-trivial one
    tensor.fuse_legs(axes=((2, 0), (1, 4), (3, 5))) gives 3 effective legs from original 6
    """
    if a.isdiag:
        raise YastError('Cannot group legs of a diagonal tensor')

    meta_fusion, order = [], []
    for group in axes:
        if isinstance(group, int):
            order.append(group)
            meta_fusion.append(a.meta_fusion[group])
        else:
            if not all(isinstance(x, int) for x in group):
                raise YastError('Inner touples of axes can only contain integers')
            order.extend(group)
            nlegs = [sum(a.meta_fusion[ii][0] for ii in group)]
            for ii in group:
                nlegs.extend(a.meta_fusion[ii])
            meta_fusion.append(tuple(nlegs))
    order = tuple(order)
    if inplace and order == tuple(ii for ii in range(a.mlegs)):
        c = a
    else:
        c = a.transpose(axes=order, inplace=inplace)
    c.meta_fusion = tuple(meta_fusion)
    c.mlegs = len(c.meta_fusion)
    return c


def unfuse_legs(a, axes, inplace=False):
    """
    Unfuse meta legs reverting one layer of fusion. Operation can be done in-place.

    New legs are inserted in place of the unfused one.

    Parameters
    ----------
    axis: int or tuple of ints
        leg(s) to ungroup.

    Returns
    -------
    tensor : Tensor
    """
    if isinstance(axes, int):
        axes = (axes,)
    c = a if inplace else a.clone()
    new_meta_fusion = []
    for ii in range(c.mlegs):
        if ii not in axes or c.meta_fusion[ii][0] == 1:
            new_meta_fusion.append(c.meta_fusion[ii])
        else:
            stack = c.meta_fusion[ii]
            lstack = len(stack)
            pos_init, cum = 1, 0
            for pos in range(1, lstack):
                if cum == 0:
                    cum = stack[pos]
                if stack[pos] == 1:
                    cum = cum - 1
                    if cum == 0:
                        new_meta_fusion.append(stack[pos_init: pos + 1])
                        pos_init = pos + 1
    c.meta_fusion = tuple(new_meta_fusion)
    c.mlegs = len(c.meta_fusion)
    return c
