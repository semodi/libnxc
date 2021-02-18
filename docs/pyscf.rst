Using Pylibnxc with PySCF
==========================

Using Pylibnxc in PySCF is as simple as changing two lines of code:

Starting from::

    from pyscf import gto
    from pyscf.dft import RKS

    mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
    mf = RKS(mol)
    mf.xc = 'PBE'
    mf.kernel()

one can use pylibnxc::

    from pyscf import gto
    from pylibnxc.pyscf import RKS

    mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
    mf = RKS(mol, nxc='GGA_PBE', nxc_kind='grid')
    mf.kernel()

The second version would run a SCF calculation using our machine-learned version of
PBE (see [Shipped functionals](#shipped-functionals).

For unrestricted Kohn-Sham calculations ```pylibnxc.pyscf.UKS`` is available as well.
The ``nxc`` keyword supports mixing of functionals similar to pyscf, e.g.
```nxc ='0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'`` would correspond to a neural network version
of PBE0.
Currently, mixing of libxc functionals with libnxc functionals is not supported.
