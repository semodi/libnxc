C++ API
================

The recommended way to use Libnxc is by linking it with Libxc. If that is however not possible, e.g. if the
electronic structure code does not support Libxc or NeuralXC functionals, the API described
in the following sections will allow developers to incorporate Libnxc into existing DFT codes.

Functional parameters
--------------------------
The struct ``func_param`` is used to set parameters that relate to the system that is being
simulated as well as the way Libnxc communicates with the electronic structure code.::

    /**
    * @param pos atomic positions
    * @param nua number of atoms
    * @param cell lattice vectors
    * @param grid number of grid points for each lattice vector
    * @param isa species index for every atom
    * @param symbols ditinct symbols
    * @param ns symbols.size()
    * @param myBox box in simulation cell (used mainly for MPI decomposition)
    * @param edens 0: return total energy, 1: return energy density  (default: 1)
    * @param add 0: set return values 1: add return values (default: 1)
    * @param cuda 0: use cpu 1: use gpu (default: 0)
    */
    struct func_param{
      double * pos; // atomic positions
      int nua; // number of atoms (pos.size())
      double * cell; // lattice vectors
      int * grid; // number of grid points for each LV
      int * isa; // species index for every atom  (relates to symbols array)
      char * symbols; //distinct symbols
      int ns; // symbols.size()
      int * myBox; // box in simulation cell (used mainly for MPI decomposition)
      int edens = defaults->edens;
      int add = defaults->add;
      int cuda = defaults->cuda;
      int gamma = defaults->gamma;
    };



Parameters ``pos`` to ``myBox`` are only relevant for atomic functionals (NeuralXC) and contain information
about the simulation box (unit cell) as well as the atomic positions and species.
``isa`` together with ``symbols`` defines the element of every atom which is relevant if the NeuralXC
functional is species dependent.

Example:::

  int * isa = {1,0,0}
  char * symbols = {'H','O'}

would be interpreted as one oxygen atoms and two hydrogen atoms ``{'O','H','H'}``.

The remaining parameters govern the evaluation of the functional:

- ``eden``:

    - ``1``: return the energy per unit particle on a grid, (default),

    - ``0``: only return the total (xc)-energy (This was mainly implemented for performance reasons)

- ``add``:

  - ``1``: adds return values for exc and the potential terms to the provided arrays,

  - ``0``: sets the return values for exc and the potential terms in the provided arrays (default)

- ``cuda``:

  - ``1``: model inference on GPUs,

  - ``0``: model inference on CPUs (default)

- ``gamma``:

  - ``1``: for GGAs and higher, the gradient of the electron density is provided,

  -  ``0``: the reduced gradient sigma is provided (default)}

The default values were chosen to closely mirror the functionality of Libxc.
If Libnxc is being used with SIESTA or CP2K, appropriate values can be set with ``nxc_set_code``::

  const int DEFAULT_CODE=0;
  const int SIESTA_GRID_CODE=1;
  const int SIESTA_ATOMIC_CODE=2;
  const int CP2K_CODE=0;

  void nxc_set_code(int code);

Note that this function has to be called *before* ``nxc_func_init`` to have any effect.


**Initializing the function**

The functional can be initialized using ``nxc_func_init``::

    const int NXC_POLARIZED=2;
    const int NXC_UNPOLARIZED=1;
    /**
    * Initializes functional
    *
    * @param[out] p loaded functional
    * @param[in] model string containing either model path or name
    * @param[in] fp functional parameters
    * @param[in, optional] nspin spin polarized/unpolarized calcuation (default NXC_UNPOLARIZED)
    */
    void nxc_func_init(nxc_func_type* p, std::string model, func_param fp, int nspin=NXC_UNPOLARIZED);


Model evaluation
------------------

Depending on which rung the loaded functional resides on one of the following methods can be used
for evaluation:::

    /**
    * Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
    * that only depend on the local density.
    *
    * @param[in] p functional to evaluate
    * @param[in] np number of grid points (size of rho)
    * @param[in] rho electron density
    * @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
    * @param[(in), out] vrho dE/drho
    */
    void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);
    void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                            double forces[], double stress[]);
    void nxc_gga_exc_vxc(nxc_func_type* p, int np, double rho[], double sigma[], double * exc, double vrho[], double vsigma[]);
    void nxc_mgga_exc_vxc(nxc_func_type* p, int np, double rho[],double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[],double vtau[]);

The arguments are defined in the same way as for Libxc with the notable exception that
sigma (and accordingly vsigma) can either be the reduced gradient or the gradient of the
density (and the corresponding potential term) depending on the parameter ``gamma`` in the ``func_param`` struct. For multidimensional arrays
the **fastest** index is understood to run over grid points. ``int np`` is the full size of the array ``rho``, i.e. for spin polarized calculations (``NXC_SPIN_POLARIZED``) it is twice the number of grid points, and equal to the number of grid points for unpolarized calculations.

NeuralXC functionals require special treatment, as their dependency on localized atomic orbitals produces additional
terms when evaluating forces and stress. These corrections can be obtained with the method ``nxc_lda_exc_vxc_fs``, which
should be called as the last step at the end of a converged SCF calcuation. When forces and stress aren't required
(e.g. during the SCF loop) it suffices to call ``nxc_lda_exc_vxc`` to evaluate the NeuralXC functional.

Other methods
-------------
We provide two methods to check the type of a functional depending on whether the functional
has already been loaded and initialized (``nxc_func_get_family``) or whether we want to
check the functional type without loading it (``nxc_func_get_family_from_path``)::

  /**
  * Check if GPU(cuda) is available
  */
  int nxc_cuda_available();
  void nxc_use_cuda(){
    defaults->useCuda();
  }

  const int LDA_TYPE=0;
  const int GGA_TYPE=1;
  const int MGGA_TYPE=2;
  const int ATOMIC_TYPE=4;
  int nxc_func_get_family(nxc_func_type* p);
  int nxc_func_get_family_from_path(std::string model);
