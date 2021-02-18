pylibnxc
==========

Loading a model
----------------
The following method loads a libnxc functional:::

  def LibNXCFunctional(name, **kwargs):
      """ Loads a Libnxc functional
      Parameters
      ----------
      name, string
          Either the name of a pre-defined functional or path to custom
          functional (will check path first and then resort to pre-defined functional)
      kind, string, optional {'grid', 'atomic'}
          default: 'grid', whether functional is grid kind (LDA, GGA etc.) or
          atomic (NeuralXC)

      Returns
      --------
      NXCFunctional
      """

Initializing a model
---------------------

If and only if the functional is of atomic (NeuralXC) kind it needs to
be initialized before usage. Keyword arguments passed to the initialize function are used to infer whether the calculation is done in periodic boundary conditions:

Periodic boundary conditions with euclidean grid:::

  def initialize(self, **kwargs):
      """Parameters
      ------------------
      unitcell, numpy.ndarray (3,3)
      	Unitcell in bohr
      grid, numpy.ndarray (3,)
      	Grid points per unitcell
      positions, numpy.ndarray (Natoms, 3)
      	atomic positions
      species, list string (Natoms)
      	atomic species (chem. symbols)
      """

Non-periodic boundary conditions with custom grid:::

  def initialize(self, **kwargs):
      """Parameters
      ------------------
      grid_coords, numpy.ndarray (Ngrid,3)
      	Grid point coordinates (in a.u.)
      grid_weights, numpy.ndarray (Ngrid,)
      	Grid point weights for integration
      positions, numpy.ndarray (N, 3)
      	atomic positions
      species, list string (N)
      	atomic species (chem. symbols)

Evaluating a model
------------------

The model can be evaluated by calling ``compute`` on a ``LibNXCFunctional`` instance:::


  def compute(self, inp, do_exc=True, do_vxc=True, **kwargs):
          """ Evaluate the functional on a given input

          Parameters
          ----------
          inp, dict of np.ndarrays
              Input electron density "rho" and its derivatives. Potential terms are
              calculated for all provided derivates. (valid keys: 'rho','sigma',
              'gamma','tau','lapl').
          do_exc, bool, optional
              no effect, only here for compatibility reasons
          do_vxc, bool, optional
              whether to compute the functional derivative(s) of the energy.
              default: True

          Returns:
          ---------
          output, dict
              Dictionary containing output values:
                  - 'zk': energy per unit particle or total energy
                  - 'vrho/vsigma/vtau' : potential terms
      """


If the functional type is "atomic" two additional keyword arguments can be provided:

  - ``do_forces``: bool, Compute the pulay force corrections. The output dict will then
  contain an entry named ``'forces'``.

  - ``edens``: bool, Return energy per unit particle if ``True``, total energy otherwise

In this case, instead of providing the electron density as ``'rho'`` the projected density
or ML-descriptors can be provided as ``'c'``. Doing so, the density projection step
is skipped but force corrections are not available. This might save resources for
codes for which analytical integrals over orbitals are available.
