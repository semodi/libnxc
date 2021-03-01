from pyscf import dft
from pyscf.lib.numpy_helper import NPArrayWithTag
from ..adapters import PySCFNXC, NXCAdapter, get_nxc_adapter
from ..functional import AtomicFunc, GridFunc, HMFunc, LibNXCFunctional
from .utils import parse_xc_code, find_max_level
from .nl import veff_mod_nl
import numpy as np
import os
from functools import partial


def KS(mol, method, nxc='', nxc_kind='grid', **kwargs):
    """ Wrapper for the pyscf RKS and UKS class
    that uses a libnxc functionals
    """
    grid_level = 0
    if 'grid_level' in kwargs:
        grid_level = kwargs.pop('grid_level')
    mf = method(mol, **kwargs)
    if grid_level:
        mf.grids.level = grid_level
    if nxc != '':
        if nxc_kind.lower() == 'atomic':
            model = get_nxc_adapter('pyscf', nxc)
            mf.get_veff = veff_mod_atomic(mf, model)
        elif nxc_kind.lower() == 'grid':
            parsed_xc = parse_xc_code(nxc)
            if '_NL' in nxc:
                mf.grids.build()
                omega = None
                for code, factor in parsed_xc[1]:
                    model = LibNXCFunctional(code, kind='grid')
                    if not omega is None:
                        if not omega == model.omega and model.omega:
                            raise Exception('ModelError: Range separation parameter' +\
                             ' omega must be consistent across models')
                    if model.omega:
                        omega = model.omega
                        
                mf.get_veff = veff_mod_nl(mf, omega,
                                     find_max_level(parsed_xc),
                                     hyb=parsed_xc[0][0],
                                     method=method)
            else:
                dft.libxc.define_xc_(mf._numint,
                                     eval_xc,
                                     find_max_level(parsed_xc),
                                     hyb=parsed_xc[0][0])
            mf.xc = nxc
        else:
            raise ValueError(
                "{} not a valid nxc_kind. Valid options are 'atomic' or 'grid'"
                .format(nxc_kind))
    return mf


RKS = partial(KS, method=dft.RKS)
UKS = partial(KS, method=dft.UKS)


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    """ Evaluation for grid-based models (not atomic)
        See pyscf documentation of eval_xc
    """
    inp = {}
    if spin == 0:
        if rho.ndim == 1:
            rho = rho.reshape(1, -1)
        inp['rho'] = rho[0]
        if len(rho) > 1:
            dx, dy, dz = rho[1:4]
            gamma = (dx**2 + dy**2 + dz**2)
            inp['sigma'] = gamma
        if len(rho) > 4:
            inp['lapl'] = rho[4]
            inp['tau'] = rho[5]
    else:
        rho_a, rho_b = rho
        if rho_a.ndim == 1:
            rho_a = rho_a.reshape(1, -1)
            rho_b = rho_b.reshape(1, -1)
        inp['rho'] = np.stack([rho_a[0], rho_b[0]])
        if len(rho_a) > 1:
            dxa, dya, dza = rho_a[1:4]
            dxb, dyb, dzb = rho_b[1:4]
            gamma_a = (dxa**2 + dya**2 + dza**2)  #compute contracted gradients
            gamma_b = (dxb**2 + dyb**2 + dzb**2)
            gamma_ab = (dxb * dxa + dyb * dya + dzb * dza)
            inp['sigma'] = np.stack([gamma_a, gamma_ab, gamma_b])
        if len(rho_a) > 4:
            inp['lapl'] = np.stack([rho_a[4], rho_b[4]])
            inp['tau'] = np.stack([rho_a[5], rho_b[5]])

    parsed_xc = parse_xc_code(xc_code)
    total_output = {'v' + key: 0.0 for key in inp}
    total_output['zk'] = 0

    for code, factor in parsed_xc[1]:
        model = LibNXCFunctional(code, kind='grid')
        output = model.compute(inp)
        for key in output:
            if output[key] is not None:
                total_output[key] += output[key] * factor

    exc, vlapl, vtau, vrho, vsigma = [total_output.get(key,None)\
      for key in ['zk','vlapl','vtau','vrho','vsigma']]

    vxc = (vrho, vsigma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc


def veff_mod_atomic(mf, model):
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    atomic potential
    """
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        rho0 = rho[:1]
        gamma = None

        exc, V_nxc = model.compute(rho0.flatten(), edens=True)

        vrho = V_nxc

        vgamma = np.zeros_like(V_nxc)
        vlapl = None
        vtau = None
        vxc = (vrho, vgamma, vlapl, vtau)
        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative

        return exc, vxc, fxc, kxc

    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mf.define_xc_(mf.xc, 'GGA')  #TODO: This doesn't seem quite right
        veff = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        mf.define_xc_(eval_xc, 'GGA')
        model.initialize(mf.grids.coords, mf.grids.weights, mol)
        vnxc = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        veff[:, :] += (vnxc[:, :] - vnxc.vj[:, :])
        veff.exc += vnxc.exc
        return veff

    return get_veff
