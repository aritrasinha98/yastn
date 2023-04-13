.. QR and SVD

Algebra
=======

Canonical form by QR
--------------------

There are different algorithms, which can bring MPS/MPO into a canonical form.
The cheapest way is by application of :meth:`QR<yastn.linalg.qr>` decomposition.

.. note::

        Both :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>`
        and :ref:`TDVP<mps/algorithms_tdvp:Time-dependent variational principle (TDVP) algorithm>`
        algorithms expect initial MPS to be canonized towards the first site.

Bring MPS/MPO into canonical form by QR decomposition

::

        # Generate random MPS with no symmetry
        psi = yamps.random_dense_mps(N=16, D=15, d=2)

        # rigth canonical form
        #
        # --A*--    --
        #   |   | =   |Identity
        # --A---    --
        #
        psi.canonize_(to='first')

        # left canonical form
        #
        #  --A*--             --
        # |  |     = Identity|
        #  --A---             --
        #
        psi.canonize_(to='last')

Check if MPS/MPO is in left/right canonical form by verifying
if each tensor forms an isometry after appropriate contraction
with its conjugate. For either left or right canonical form

.. literalinclude:: /../../tests/mps/test_canonize.py
        :pyobject: check_canonize

Canonical form by SVD
---------------------

Bringing MPS/MPO into canonical form through :meth:`SVD<yastn.linalg.svd>` decomposition
is computationally more expensive than QR, but allows for struncation.
Truncation is governed by options passed as :code:`opts_dict`
to :meth:`yastn.linalg.truncation_mask` that is applied after each SVD during the sweep through MPS/MPO.

.. note::
        Faithfull SVD truncation requires MPS/MPO to be in the canonical form
        of the opposite direction to the direction of the truncation sweep.

::

        # There are different options which we can pass, see yastn.linalg.svd.
        # Defaults are assumed for options not explictly specified.
        #
        opts_svd = {
                'D_total': 4,      # total number of singular values to keep
                'D_block': 2,      # maximal number of singular values to keep in a single block
                'tol': 1e-6,       # relative tolerance of singular values below which to
                                   # truncate across all blocks
                'tol_blocks': 1e-6 # relative tolerance of singular values below which to
                                   # truncate within individual blocks
        }

        # Generate random MPS with no symmetry
        #
        psi = yamps.random_dense_mps(N=16, D=15, d=2)

        # Bring MPS to canonical form and truncate (here, right canonical form).
        # For MPS we usually normalize the state.
        #
        psi.canonize_(to='last')
        psi.truncate_(to='first', opts_svd=opts_svd)

        # Generate random MPO with no symmetry
        #
        H= generate_random.mpo_random(config_dense, N=16, Dmax=25, d=2, d_out=2)

        # Bring MPO to canonical form and truncate (here, left canonical form).
        # Note: for MPO we do not want to change overall scale, thus no normalization.
        #
        H.canonize_(to='first', normalize=False)
        H.truncate_(to='last', opts_svd=opts_svd, normalize=False)

Multiplication
--------------

We can test the multiplication of MPO and MPS using an example of a ground state obtained with :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>`.

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: test_multiplication
