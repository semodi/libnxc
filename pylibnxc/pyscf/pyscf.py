from pyscf import dft
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


def veff_mod_rad(mf, model) :
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    potential
    """
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        rho0 = rho[:1]
        gamma = None

        exc, V_nxc = model.compute(rho0.flatten(),edens=True)

        # exc = exc/rho0.flatten()
        # exc = exc/model.grid_weights
        # exc /= len(model.grid_weights)

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
        veff = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        mf.define_xc_(eval_xc,'GGA')
        model.initialize(mf.grids.coords, mf.grids.weights, mol)
        vnxc = dft.rks.get_veff(mf, mol, dm_val, dm_last,
            vhf_last, hermi)
        veff[:, :] += (vnxc[:, :] - vnxc.vj[:,:])
        veff.exc += vnxc.exc
        return veff
    return get_veff
