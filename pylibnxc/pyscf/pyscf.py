from pyscf import dft
from pyscf.lib.numpy_helper import NPArrayWithTag
from ..adapters import PySCFNXC, NXCAdapter
from ..functional import AtomicFunc, GridFunc, HMFunc, LibNXCFunctional
import numpy as np
import os
def RKS(mol, nxc='', **kwargs):
    """ Wrapper for the pyscf RKS (restricted Kohn-Sham) class
    that uses a NeuralXC potential
    """
    mf = dft.RKS(mol, **kwargs)
    if not nxc is '':
        if os.path.exists(nxc):
            model = get_nxc_adapter(nxc)
            mf.get_veff = veff_mod_atomic(mf, model)
        else:
            dft.libxc.define_xc_(mf._numint, eval_xc, nxc.split('_')[1])
            mf.xc = nxc
    return mf


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    inp = {}
    if rho.ndim == 1:
        rho = rho.reshape(1,-1)
    if spin == 0:
        inp['rho'] = rho[0]
        if len(rho) > 1:
            dx, dy, dz = rho[1:4]
            gamma = (dx**2 + dy**2 + dz**2)
            inp['sigma'] = gamma
        if len(rho) > 4:
            inp['lapl'] = rho[4]
            inp['tau'] = rho[5]
    else:
        raise NotImplementedError('Spin polarized not implemented yet')

    model = LibNXCFunctional(name=xc_code, kind='hm')
    output = model.compute(inp)

    exc = output.get('zk', None)
    vlapl = output.get('vlapl', None)
    vtau = output.get('vtau', None)
    vrho = output.get('vrho', None)
    vgamma = output.get('vsigma', None)
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc

def veff_mod_atomic(mf, model) :
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    potential
    """
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        rho0 = rho[:1]
        gamma = None

        exc, V_nxc = model.compute(rho0.flatten(),edens=True)

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
        vnxc = dft.rks.get_veff(mf, mol, dm, dm_last,
            vhf_last, hermi)
        veff[:, :] += (vnxc[:, :] - vnxc.vj[:,:])
        veff.exc += vnxc.exc
        return veff
    return get_veff
