import pyscf
from pyscf import gto, dft
from pyscf.dft import RKS
from pyscf.scf import hf, RHF, RKS
from pyscf.scf.chkfile import load_scf
from pyscf.lib.numpy_helper import NPArrayWithTag
from ..adapters import get_nxc_adapter
import numpy as np

def RKS(mol, nxc='', **kwargs):
    """ Wrapper for the pyscf RKS (restricted Kohn-Sham) class
    that uses a NeuralXC potential
    """
    mf = dft.RKS(mol, **kwargs)
    if not nxc is '':
        model = get_nxc_adapter('pyscf', nxc)
        mf.get_veff = veff_mod_rad(mf, model)
    return mf


def compute_KS(atoms, path='pyscf.chkpt', basis='ccpvdz', xc='PBE', nxc='', approx_val=False):
    """ Given an ase atoms object, run a pyscf RKS calculation on it and
    return the results
    """
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]

    mol = gto.M(atom=mol_input, basis=basis)
    if approx_val:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.kernel()
        mf.mo_occ[1:] = 0
        dm_core = mf.make_rdm1()

    mf = RKS(mol, nxc=nxc)

    if approx_val:
        mf.dm_core = dm_core
    mf.set(chkfile=path)
    mf.xc = xc
    mf.kernel()
    return mf, mol

def veff_mod_rad(mf, model) :
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    potential
    """
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        rho0 = rho[:1]
        gamma = None

        exc, V_nxc = model.compute(rho0.flatten())

        exc = exc/rho0.flatten()
        exc = exc/model.grid_weights
        exc /= len(model.grid_weights)

        vrho = V_nxc

        vgamma = np.zeros_like(V_nxc)
        vlapl = None
        vtau = None
        vxc = (vrho , vgamma, vlapl, vtau)
        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative

        return exc, vxc, fxc, kxc

    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mf.define_xc_(mf.xc,'GGA')
        veff = pyscf.dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        mf.define_xc_(eval_xc,'GGA')
        if hasattr(mf,'dm_core'):
            dm_val = dm - mf.dm_core
        else:
            dm_val = dm

        model.initialize(mf.grids.coords,mf.grids.weights, mol)
        vnxc = pyscf.dft.rks.get_veff(mf, mol, dm_val, dm_last,
            vhf_last, hermi)
        veff[:, :] += (vnxc[:, :] - vnxc.vj[:,:])
        veff.exc += vnxc.exc

        return veff
    return get_veff
