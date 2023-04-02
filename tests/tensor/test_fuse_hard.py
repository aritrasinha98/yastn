""" Test elements of fuse_legs(... mode='hard') """
import numpy as np
import unittest
import pytest
import yastn
try:
    from .configs import config_U1, config_dense, config_Z2xU1
except ImportError:
    from configs import config_U1, config_Z2xU1, config_dense

tol = 1e-10  #pylint: disable=invalid-name


class FusionSyntax(unittest.TestCase):
    
    def test_fuse_hard(self):
        # define a rank-5 U(1)-symmetric tensor
        a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                      t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                      D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

        # this tensor has 10 non-zero blocks, the largest one being
        # of shape (2, 4, 6, 8, 9) with total number of 3456 elements
        a.print_blocks_shape()
        #
        # charges         shape
        # (0, 0, 0, 0, 0) (1, 3, 5, 7, 9)
        # (0, 0, 0, 1, 1) (1, 3, 5, 8, 10)
        # (0, 0, 1, 1, 0) (1, 3, 6, 8, 9)
        # (0, 1, 0, 1, 0) (1, 4, 5, 8, 9)
        # (1, 0, 0, 0, 1) (2, 3, 5, 7, 10)
        # (1, 0, 1, 0, 0) (2, 3, 6, 7, 9)
        # (1, 0, 1, 1, 1) (2, 3, 6, 8, 10)
        # (1, 1, 0, 0, 0) (2, 4, 5, 7, 9)
        # (1, 1, 0, 1, 1) (2, 4, 5, 8, 10)
        # (1, 1, 1, 1, 0) (2, 4, 6, 8, 9)

        # Lets fuse last three legs of the tensor a into a new leg
        b = a.fuse_legs(axes=(0, 1, (2, 3, 4)), mode='hard')

        # resulting tensor has just four non-zero blocks, with the largest
        # one holding 9176 elements
        b.print_blocks_shape()
        #
        # (0, 0, 0) (1, 3, 1147)
        # (0, 1, -1) (1, 4, 360)
        # (1, 0, 1) (2, 3, 1208)
        # (1, 1, 0) (2, 4, 1147)

        # We can also fuse more than a single group of indices.
        # Their order can be permuted as well
        c0 = a.fuse_legs(axes=(0, (3, 4), (2, 1)), mode='hard')

        # Fusion can be applied successively - fusing fused spaces together.
        # This results in a rank-2 tensor, equivalent to block-sparse matrix
        c1 = c0.fuse_legs(axes=((0, 1), 2), mode='hard')

        # Resulting matrix has three blocks and the largest block holds 13604 elements
        assert c1.s == (-1, 1)
        c1.print_blocks_shape()
        #
        # (0, 0) (283, 15)
        # (1, 1) (358, 38)
        # (2, 2) (144, 24)
        
        # The fusion can be simply reverted, proceeding step-by-step in reverse
        # order of the applied fusions. 
        # NOTE: Unfusing an index *does not* permute the resulting indices into
        #       into their original order
        #
        # fusion step 1: 0 1 2 3 4 -> permute -> 0 3 4 2 1 -> fuse -> 0 (3 4) (2 1) = 0 1 2
        # fusion step 2: 0 1 2 -> fuse -> (0 1) 2 = 0 1 
        #
        # unfuse step 2: 0 1 -> unfuse -> (0 1) 1->2 = 0 1 2
        c0_0 = c1.unfuse_legs(axes=0)
        assert yastn.norm(c0_0 - c0) < tol

        # unfuse step 1: 0 1 2 -> unfuse -> 0 (3->1 4->2) (2->3 1->4) = 0 1 2 3 4
        # 
        # Hence, to retrieve original tensor, we have to permute its indices
        a_0 = c0_0.unfuse_legs(axes=(1, 2))
        a_0 = a_0.transpose(axes=(0, 4, 3, 1, 2))
        assert yastn.norm(a - a_0) < tol


def test_hard_split():
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)), mode='hard')
    af = af.fuse_legs(axes=((0, 1), 2), mode='hard')
    Uf, Sf, Vf = yastn.linalg.svd(af, axes=(0, 1))

    U, S, V = yastn.linalg.svd(a, axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs(axes=(0, (2, 1), 3), mode='hard')
    U = U.fuse_legs(axes=((0, 1), 2), mode='hard')
    V = V.fuse_legs(axes=(0, (1, 2)), mode='hard')

    US = yastn.tensordot(U, S, axes=(1, 0))
    a2 = yastn.tensordot(US, V, axes=(1, 0))
    assert yastn.norm(af - a2) < tol  # == 0.0
    USf = yastn.tensordot(Uf, Sf, axes=(1, 0))
    a3 = yastn.tensordot(USf, Vf, axes=(1, 0))
    assert yastn.norm(af - a3) < tol  # == 0.0
    a3 = a3.unfuse_legs(axes=0)
    a3 = a3.unfuse_legs(axes=(1, 2)).move_leg(source=2, destination=1)
    assert yastn.norm(a - a3) < tol  # == 0.0

    Qf, Rf = yastn.linalg.qr(af, axes=(0, 1))
    Q, R = yastn.linalg.qr(a, axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3), mode='hard')
    Q = Q.fuse_legs(axes=((0, 1), 2), mode='hard')
    assert yastn.norm(Q - Qf) < tol  # == 0.0
    Rf = Rf.unfuse_legs(axes=1)
    assert yastn.norm(R - Rf) < tol  # == 0.0

    aH = yastn.tensordot(af, af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = yastn.linalg.eigh(aH, axes=(0, 1))
    Uf = Uf.unfuse_legs(axes=0)
    UVf = yastn.tensordot(Uf, Vf, axes=(2, 0))
    aH2 = yastn.tensordot(UVf, Uf, axes=(2, 2), conj=(0, 1))
    aH = aH.unfuse_legs(axes=(0, 1))
    assert yastn.norm(aH2 - aH) < tol  # == 0.0


def test_hard_transpose():
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    assert a.get_shape() == (3, 5, 7, 9, 11, 13)

    b = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')
    assert b.get_shape() == (15, 7, 99, 13)

    c = np.transpose(b, axes=(3, 2, 1, 0))
    assert c.get_shape() == (13, 99, 7, 15)

    c = c.unfuse_legs(axes=(1, 3))
    assert c.get_shape() == (13, 9, 11, 7, 3, 5)

    c = b.move_leg(source=1, destination=2)
    assert c.get_shape() == (15, 99, 7, 13)

    c = c.unfuse_legs(axes=(1, 0))
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_hard_dot():
    """ integration of hard fusion with dot """
    # Z2 x U1
    legs_a = [yastn.Leg(config_Z2xU1, s=-1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D=(1, 2, 2, 4)),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D= (9, 4, 3, 2)),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(7, 8, 9, 10)),
            yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(5, 6, 7, 8)),
            yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(1, 2, 2, 4))]
    a = yastn.rand(config=config_Z2xU1, legs=legs_a)

    legs_b = [legs_a[n].conj() for n in (0, 1, 4, 3)]
    b = yastn.rand(config=config_Z2xU1, legs=legs_b)

    aa = yastn.fuse_legs(a, axes=((0, 3), (4, 1), 2), mode='hard')
    bb = yastn.fuse_legs(b, axes=((0, 3), (2, 1)), mode='hard')

    c = yastn.tensordot(a, b, axes=((0, 1, 3, 4), (0, 1, 3, 2)))
    cc = yastn.tensordot(aa, bb, axes=((0, 1), (0, 1)))
    assert yastn.norm(c -  cc) < tol

    aaa = yastn.unfuse_legs(aa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yastn.norm(a - aaa) < tol
    bbb = yastn.unfuse_legs(bb, axes=0)
    bbb = yastn.unfuse_legs(bbb, axes=2).transpose(axes=(0, 3, 2, 1))
    assert yastn.norm(b - bbb) < tol

    aa = yastn.fuse_legs(aa, axes=(0, (1, 2)), mode='hard')
    aa = yastn.fuse_legs(aa, axes=[(0, 1)], mode='hard')
    aaa = yastn.unfuse_legs(aa, axes=0)
    aaa = yastn.unfuse_legs(aaa, axes=1)
    aaa = yastn.unfuse_legs(aaa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yastn.norm(a - aaa) < tol

    # U1
    legs_a = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(1, 2, 3)),
              yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
              yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
              yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yastn.rand(config=config_U1, legs=legs_a)

    legs_b = [legs_a[0].conj(),
              legs_a[1].conj(),
              yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(10, 7, 11))]
    b = yastn.rand(config=config_U1, legs=legs_b)

    bb = yastn.fuse_legs(b, axes=((0, 1), 2), mode='hard')
    aa =  yastn.fuse_legs(a, axes=((0, 1), 2, 3), mode='hard')

    aaa = yastn.unfuse_legs(aa, axes=0)
    bbb = yastn.unfuse_legs(bb, axes=0)

    c = yastn.tensordot(a, b, axes=((0, 1), (0, 1)))
    cc = yastn.tensordot(aa, bb, axes=(0, 0))

    assert yastn.norm(c -  cc) < tol
    assert yastn.norm(a - aaa) < tol
    assert yastn.norm(b - bbb) < tol


def test_hard_dot_sparse():
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 5, 3), val='rand')
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 6, 3), val='rand')
    a.set_block(ts=(1, 2, -1, 2), Ds=(1, 2, 6, 4), val='rand')

    b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 4, 4), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 3, 4), val='rand')
    b.set_block(ts=(2, 2, 1, 0), Ds=(2, 2, 3, 4), val='rand')

    aa = yastn.fuse_legs(a, axes=((1, 0), 2, 3), mode='hard')
    bb = yastn.fuse_legs(b, axes=((1, 0), 2, 3), mode='hard')

    leg = aa.get_legs(0)
    xx = yastn.rand(config=aa.config, legs=[leg, leg.conj()])

    yastn.tensordot(xx, aa, axes=(1, 0))
    yastn.tensordot(xx, aa, axes=(0, 0), conj = (1, 0))

    c = yastn.tensordot(a, b, axes=((0, 1), (0, 1)), conj=(1, 0))
    cc = yastn.tensordot(aa, bb, axes=(0, 0), conj=(1, 0))
    assert yastn.norm(c -  cc) < tol

    aat = aa.fuse_legs(axes=((1, 2), 0), mode='hard').conj()
    bbt = bb.fuse_legs(axes=(0, (1, 2)), mode='hard')
    ccc = yastn.tensordot(aat, bbt, axes=(1, 0))
    assert yastn.norm(c -  ccc.unfuse_legs(axes=(0, 1))) < tol

    aaa = yastn.unfuse_legs(aa, axes=0).transpose(axes=(1, 0, 2, 3))
    bbb = yastn.unfuse_legs(bb, axes=0).transpose(axes=(1, 0, 2, 3))
    assert yastn.norm(a - aaa) < tol
    assert yastn.norm(b - bbb) < tol


def _test_fuse_mix(a):
    ma = a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)), mode='meta')
    assert (ma.ndim_n, ma.ndim) == (6, 3)
    ha = a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)), mode='hard')
    assert (ha.ndim_n, ha.ndim) == (3, 3)

    hma = ma.fuse_legs(axes=((2, 0), 1), mode='hard')
    assert (hma.ndim_n, hma.ndim) == (2, 2)
    hha = ha.fuse_legs(axes=((2, 0), 1), mode='hard')
    assert (hha.ndim_n, hha.ndim) == (2, 2)
    mma = ma.fuse_legs(axes=((2, 0), 1), mode='meta')
    assert (mma.ndim_n, mma.ndim) == (6, 2)
    mha = ha.fuse_legs(axes=((2, 0), 1), mode='meta')
    assert (mha.ndim_n, mha.ndim) == (3, 2)

    assert yastn.norm(hma - hha) < tol

    fmma = yastn.fuse_meta_to_hard(mma)
    fmha = yastn.fuse_meta_to_hard(mha)
    fhha = yastn.fuse_meta_to_hard(hha)
    assert yastn.norm(fmma - hha) < tol
    assert yastn.norm(fmha - hha) < tol
    assert yastn.norm(fhha - hha) < tol


def test_fuse_mix():
    a = yastn.randR(config=config_U1, s=(1, -1, 1, 1, -1, 1),
                    t=[(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)],
                    D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    _test_fuse_mix(a)

    a = yastn.Tensor(config=config_U1, s=(1, -1, 1, 1, -1, 1))
    a.set_block(ts=(1, 2, -1, 2, 0, 0), Ds=(1, 2, 3, 4, 5, 6), val='rand')
    a.set_block(ts=(2, 1, 1, -2, 1, 1), Ds=(6, 5, 4, 3, 2, 1), val='rand')
    _test_fuse_mix(a)


def test_auxliary_merging_functions():
    mf1 = (1,)
    nt = yastn.tensor._merging._mf_to_ntree(mf1)
    mfx = yastn.tensor._merging._ntree_to_mf(nt)
    assert mf1 == mfx
    yastn.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf1 = tuple(yastn.tensor._merging._ntree_to_mf(nt))
    assert new_mf1 == (1,)

    mf2 = (9, 5, 1, 3, 2, 1, 1, 1, 1, 4, 1, 3, 1, 1, 1)
    nt = yastn.tensor._merging._mf_to_ntree(mf2)
    mfx = yastn.tensor._merging._ntree_to_mf(nt)
    assert mf2 == mfx
    yastn.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf2 = tuple(yastn.tensor._merging._ntree_to_mf(nt))
    assert new_mf2 == (6, 4, 1, 2, 1, 1, 1, 2, 1, 1)

    mf3 = (8, 2, 1, 1, 1, 5, 3, 1, 1, 1, 1, 1)
    nt = yastn.tensor._merging._mf_to_ntree(mf3)
    mfx = yastn.tensor._merging._ntree_to_mf(nt)
    assert mf3 == mfx
    yastn.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf3 = tuple(yastn.tensor._merging._ntree_to_mf(nt))
    assert new_mf3 == (5, 1, 1, 3, 1, 1, 1)

    axes, new_mfs = yastn.tensor._merging._consume_mfs_lowest((mf1, mf2, mf3))
    assert axes == ((0,), (1,), (2, 3), (4,), (5,), (6,), (7, 8, 9), (10, 11), (12,), (13, 14, 15), (16,), (17,))
    assert (new_mf1, new_mf2, new_mf3) == new_mfs


def test_fuse_hard_dense():
    # for dense
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(6, 2, 6, 2), dtype='float64')
    af = yastn.fuse_legs(a, axes=((1, 2), (3, 0)), mode='hard')
    tra = yastn.trace(a, axes=((1, 2), (3, 0)))
    traf = yastn.trace(af, axes=(0, 1))
    assert yastn.norm(tra - traf) < tol


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_transpose_and_merge_backward():
    import torch
    # U1
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(3, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(3, 11, 12))]
    a = yastn.rand(config=config_U1, legs=legs)

    b = yastn.fuse_legs(a, axes=((0, 1), 2, 3), mode='hard')

    target_block = (1, 1, -1, -1)
    target_block_size = a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        tmp_a = yastn.fuse_legs(a, axes=((0, 1), 2, 3), mode='hard')
        ab = yastn.vdot(b, tmp_a)
        return ab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(),requires_grad=True),)
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_unmerge_backward():
    import torch
    # U1
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(3, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(3, 11, 12))]
    a = yastn.rand(config=config_U1, legs=legs)

    b = yastn.fuse_legs(a, axes=(0,(1,2),3), mode='hard')

    target_block = (1, 1, -1, -1)
    target_block_size = a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        tmp_a = yastn.fuse_legs(a, axes=((0, 1, 2), 3), mode='hard')
        tmp_a = yastn.unfuse_legs(tmp_a, axes=0)
        tmp_a = yastn.fuse_legs(tmp_a, axes=(0, (1, 2), 3), mode='hard')
        ab = yastn.vdot(b, tmp_a)
        return ab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test


def test_leg_outer_product():
    l0 = yastn.Leg(config_Z2xU1, s=-1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D=(1, 2, 2, 4))
    l1 = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(7, 8, 9, 10))
    l2 = yastn.Leg(config_Z2xU1, s=1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D= (9, 4, 3, 2))
    l3 = yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(5, 6, 7, 8))

    a = yastn.rand(config=config_Z2xU1, legs=[l0, l1, l2, l3])

    fa = yastn.fuse_legs(a, axes=((0, 1), (2, 3)), mode='hard')
    pfa0 = yastn.leg_outer_product(l0, l1)
    lfa0 = fa.get_legs(axes=0)
    pfa1 = yastn.leg_outer_product(l2, l3)
    lfa1 = fa.get_legs(axes=1)
    assert (pfa0, pfa1) == (lfa0, lfa1)

    ffa = yastn.fuse_legs(fa, axes=[(0, 1)], mode='hard')
    pffa = yastn.leg_outer_product(lfa0, lfa1, t_allowed=[(0, 0)])
    lffa = ffa.get_legs(axes=0)
    assert pffa == lffa
    assert pffa.is_fused()

    ul0, ul1 = yastn.leg_undo_product(lfa0)
    ul2, ul3 = yastn.leg_undo_product(lfa1)
    ulf0, ulf1 = yastn.leg_undo_product(lffa)
    assert (ul0, ul1, ul2, ul3) == (l0, l1, l2, l3)
    assert (ulf0, ulf1) == (lfa0, lfa1)
    assert not ul0.is_fused()


if __name__ == '__main__':
    unittest.main()
    test_leg_outer_product()
    test_fuse_hard_dense()
    test_hard_split()
    test_hard_transpose()
    test_hard_dot()
    test_hard_dot_sparse()
    test_fuse_mix()
    test_auxliary_merging_functions()
    # test_transpose_and_merge_backward()
    # test_unmerge_backward()
