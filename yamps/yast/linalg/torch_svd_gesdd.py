'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(self, A, ad_decomp_reg):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V, ad_decomp_reg)
        return U, S, V
    
    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V, ad_decomp_reg = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = S.size(0)

        F = (S - S[:, None])
        F = safe_inverse(F,epsilon=ad_decomp_reg)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G = safe_inverse(G,epsilon=ad_decomp_reg)
        G.diagonal().fill_(0)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)

        return dA, None

def test_SVDSYMEIG_random():
    M, N = 50, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD.apply, A, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    test_SVDSYMEIG_random()