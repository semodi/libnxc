Installation
=========================================================

Dependencies
----------------

Libnxc depends on PyTorch. The C++/Fortran implementation requires that libtorch is made available at compile time, pylibnxc depends on pytorch.
Both libraries are available for download on the `pytorch website <https://pytorch.org/get-started/locally/>`_.

pylibnxc further depends on numpy, which is a requirement for pytorch and should therefore be installed automatically.

**Optional dependencies:**

For unit testing, pylibnxc currently requires `pyscf <https://sunqm.github.io/pyscf/install.html>`_ which can be obtained through::

    pip install pyscf

To unit test the C++/Fortran implementation `GoogleTest <https://github.com/google/googletest>`_ is required.

Installation
---------------
**To compile Libnxc:**

#. Create a work directory and copy ``utils/Makefile`` into it (this can be done quickly with ``sh config.sh``)
#. Adjust arch.make according to your system
#. Change into the work directory and run ``make``. The default target is `make all` which will build the library ``libnxc.so``.
If ``LIBXCDIR`` is set in ``arch.make``, ``make all`` will create the necessary symbolic links in your Libxc installation

**To install pylibnxc:**

pylibnxc operates independent of Libnxc and can be installed from within the root directory with::

    pip install -e .
