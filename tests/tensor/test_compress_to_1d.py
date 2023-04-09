""" fill_tensor (which is called in: rand, zeros, ones), yastn.to_numpy """
import numpy as np
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def run_simple_compress(a):
    r1d, meta = a.compress_to_1d()
    b = yastn.decompress_from_1d(r1d, meta=meta)
    assert yastn.norm(a - b) < tol


def run_compress_wit_meta(a, a_reference) :
    _, meta = a_reference.compress_to_1d()  # read meta here
    r1d, _ = a.compress_to_1d(meta)  # use meta here
    c = yastn.decompress_from_1d(r1d, meta=meta)
    assert yastn.norm(a - c) < tol


def test_compress_to_1d_basic():
    # 3d dense
    a = yastn.rand(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    run_simple_compress(a)
    # 0d U1
    a = yastn.ones(config=config_U1) # s=() # t=(), D=()
    run_simple_compress(a)
    # diagonal Z2xU1
    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, -1), (1, 0), (0, 1)), D=(2, 3, 4))
    a = yastn.rand(config=config_Z2xU1, isdiag=True, legs=leg)
    run_simple_compress(a)
    run_compress_wit_meta(a, 2 * a)



def test_compress_to_1d_embed():
    """ test embedding zeros to match another tensor """
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    a.set_block(ts=(0, 1, 0, -1), Ds=(5, 6, 7, 8))
    # creates tensor matching a filling in all possible matching blocks with zeros.
    a_reference = yastn.zeros(config=a.config, legs=a.get_legs())
    # here have to add zero blocks
    run_compress_wit_meta(a, a_reference)

    # here have to embed to match hard-fusions
    af = a.fuse_legs(axes=((0, 1), (2, 3)))
    af_reference = a_reference.fuse_legs(axes=((0, 1), (2, 3)))
    run_compress_wit_meta(af, af_reference)


def test_compress_to_1d_exceptions():
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    _, a_meta = a.compress_to_1d()

    ad = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3)))
    _, ad_meta = ad.compress_to_1d()

    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, s=(1, 1, 1, 1))
        _ = b.compress_to_1d(a_meta)
        # Tensor signature do not match meta.
    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1), n=1)
        _ = b.compress_to_1d(a_meta)
        # Tensor charge than do not match meta.
    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, s=(1, -1))
        _ = b.compress_to_1d(ad_meta)
        # Tensor diagonality do not match meta.
    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b = b.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        _ = b.compress_to_1d(a_meta)
        # Tensor meta-fusion structure do not match meta.
    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        _ = b.compress_to_1d(a_meta)
        # Tensor has blocks that do not appear in meta.
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _, af_meta = af.compress_to_1d()
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = bf.compress_to_1d(af_meta)
        # Tensor fused legs do not match metadata.


if __name__ == '__main__':
    test_compress_to_1d_basic()
    test_compress_to_1d_embed()
    test_compress_to_1d_exceptions()
