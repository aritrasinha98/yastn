import numpy as np
import yast.tn.peps as peps

def initialize_peps_purification(fid, net):
    """
    initialize peps tensors into infinite-temeprature state, 
    fid is identity operator in local space with desired symmetry
    """
    
    A = fid / np.sqrt(fid.get_shape(1)) 
    A = A.fuse_legs(axes=[(0, 1)])            
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
   
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)
    Gamma._data = {ms: A for ms in net.sites()}
    return Gamma