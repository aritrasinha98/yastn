""" Define rules for Z2 symmetry"""
import numpy as np

name = 'Z2'

nsym = 1  # single int is used to distinguish symmetry sectors

def fuse(charges, signatures, new_signature):
    """
    Fusion rule for Z2 symmetry

    Parameters
    ----------
        charges: nparray(int)
            k x number_legs x nsym matrix, where k is the number of independent blocks.

        signatures: nparray(ind)
            vector with number_legs elements

        new_signature: int

    Returns
    -------
        teff: nparray(int)
            matrix of effective fused charges of size k x nsym for new signature
    """
    return np.mod(new_signature * (charges.swapaxes(1,2) @ signatures), 2)
