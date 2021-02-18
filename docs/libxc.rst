Using Libnxc with Libxc
===========================

The simplest way to use machine learned functionals in existing electronic
structure codes is by linking Libxc with Libnxc. This way any code that supports
Libxc functionals (Quantum Espresso, CP2K, Vasp, Psi4, ...) has access to Libnxc routines.
This comes with two caveats:

  1. Only grid based functionals are supported through this solution. There is currently
  no way to pass the additional information needed by NeuralXC functional (atomic positions, unit cell size etc.)
  through Libxc.

  2. Without adjusting the electronic structure code, using Libnxc with Neural Network functionals
  may be significantly slower than calling traditional functionals. This is because some codes pass
  the density grid point by grid point. This is fine for traditional functionals, but to efficiently
  evaluate NN functionals, grid points should be processed in a vectorized fashion. Codes like
  Quantum espresso already do so, but others, like CP2K, will need to be modified to ensure efficient
  evaluation.

Using Libnxc from within Libxc is straightforward:

- Set the ``LIBXCDIR`` variable in arch.make to the location of the libxc ``src`` directory.

- In order to be able to run unit tests using Libxc ``LIBXC_INCLUDE`` and ``LIBXC_LD`` need to be set accordingly.

- From within the build directory ``make libxc`` will create symbolic links of the required
  Libnxc source files within the Libxc ``src`` directory and copy a modified ``Makefile.am`` that includes
  instructions to link to Libnxc. In some cases automake needs to be run in the Libxc directory.

- After successfully compiling Libnxc, Libxc should be re-compiled to include the new functionals.
