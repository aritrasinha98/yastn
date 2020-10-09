import numpy as np
import torch
import unittest
import yamps.tensor as TA
import settings_full
import settings_U1
import settings_U1_U1
import yamps.peps.config as cfg
from yamps.peps.ipeps.ipeps_abelian import IPEPS_ABELIAN
from yamps.peps.ctm.generic_abelian.env_abelian import ENV_ABELIAN
import yamps.peps.ctm.generic_abelian.ctmrg as ctmrg_abelian
from yamps.peps.ctm.generic.env import ENV, init_env
import yamps.peps.ctm.generic.ctmrg as ctmrg

class Test_env_abelian(unittest.TestCase):
    
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_full(cls):
        a = TA.rand(settings=settings_full, s=cls._ref_s_dir, D=(2, 3, 2, 3, 2))
        b = TA.rand(settings=settings_full, s=cls._ref_s_dir, D=(2, 3, 2, 3, 2))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_full, sites, vertexToSite)

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls):

        # AFM D=2
        a = TA.zeros(settings=settings_U1, s=cls._ref_s_dir, n=0,
                        t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val='ones')
        tmp_B= 0.3*np.ones((1,1,1,1,1))
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp_B)

        b = TA.zeros(settings=settings_U1, s=cls._ref_s_dir, n=0,
                        t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        b.set_block((0,0,0,0,0), (1,1,1,1,1), val='ones')
        b.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp_B)

        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    @classmethod
    def _get_2x1_BIPARTITE_U1_U1(cls):
        a = TA.rand(settings=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
                        t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
                        D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

        b = TA.rand(settings=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
                        t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
                        D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1_U1, sites, vertexToSite)

    def setUp(self):
        pass


    def test_ctmrg_abelian_full_chi1(self):
        state= self._get_2x1_BIPARTITE_full()
        env= ENV_ABELIAN(state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f)

    def test_ctmrg_abelian_full(self):
        state= self._get_2x1_BIPARTITE_full()
        env= ENV_ABELIAN(chi=8, state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= c.split_svd((0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=50
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        state_dense= state.to_dense()
        state_dense.sites= {sid: torch.from_numpy(s) for sid,s in state_dense.sites.items()}
        env_dense= env_out.to_dense()
        env_dense.C= {cid: torch.from_numpy(c) for cid,c in env_dense.C.items()}
        env_dense.T= {tid: torch.from_numpy(t) for tid,t in env_dense.T.items()}
        
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v=torch.svd(c, compute_uv=False)
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter= 50
        env_out, *ctm_log= ctmrg.run(state_dense, env_dense, conv_check=ctmrg_conv_f,
            ctm_args=cfg.ctm_args)


    def test_ctmrg_abelian_U1_chi1(self):
        state= self._get_2x1_BIPARTITE_U1()
        env= ENV_ABELIAN(state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f)

    def test_ctmrg_abelian_U1(self):
        chi=9
        np.random.seed(2)
        state= self._get_2x1_BIPARTITE_U1()
        env= ENV_ABELIAN(chi=chi, state=state, init=True)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= c.split_svd((0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        print("----- CTMRG_ABELIAN FINISHED -----")
        cfg.global_args.dtype= torch.float64

        state_dense= state.to_dense()
        state_dense.sites= {sid: torch.from_numpy(s) for sid,s in state_dense.sites.items()}
        state_dense.write_to_file("TEST_state.json")
        # env_dense= env_out.to_dense()
        # env_dense.C= {cid: torch.from_numpy(c) for cid,c in env_dense.C.items()}
        # env_dense.T= {tid: torch.from_numpy(t) for tid,t in env_dense.T.items()}
        env_dense= ENV(chi, state=state_dense, global_args=cfg.global_args)
        init_env(state_dense, env_dense)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v=torch.svd(c, compute_uv=False)
                print(f"{cid}: {s}")
            return False, history

        env_out, *ctm_log= ctmrg.run(state_dense, env_dense, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

if __name__ == '__main__':
    #unittest.main()
    T= Test_env_abelian()
    #T.test_ctmrg_abelian_full_chi1()
    #T.test_ctmrg_abelian_full()

    #T.test_ctmrg_abelian_U1_chi1()
    T.test_ctmrg_abelian_U1()