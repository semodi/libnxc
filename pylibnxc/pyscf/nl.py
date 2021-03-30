from .utils import parse_xc_code, find_max_level, get_v_aux
from pyscf import df
import scipy
import numpy as np
from opt_einsum import contract
from pyscf import dft
from ..functional import LibNXCFunctional



def veff_mod_nl(mf, omega, max_level, hyb, method):
    """ Effective potential for functionals involving the (screened) coulomb
    potential as input
    """
    def block_loop_decorator(func):
        """ Decorator for the numint.block_loop generator.
        numint.block_loop splits grid into blocks that fit in memory. This decorator is
        used to keep track of the block indices and store it to the mf object.
        """
        def block_loop(*args, **kwargs):
            # Equip weights with index column to track block size
            grids = args[1]
            weights = grids.weights
            grids.weights = np.stack([weights, np.arange(len(weights))], axis = -1)
            for ao, mask, weight, coords in func(*args,**kwargs):
                mf.mask_indices = weight[:, 1].astype(int)
                yield ao, mask, weight[:,0], coords
            grids.weights = grids.weights[:, 0]
        return block_loop

    mol = mf.mol
    auxbasis = df.addons.make_auxbasis(mol, mp2fit=False)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    df_3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
    df_2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
    df_2c_inv = scipy.linalg.pinv(df_2c)
    if not isinstance(omega, list):
        omega = [omega]
    vh_on_grid = np.stack([get_v_aux(mf.grids.coords, auxmol,o) for o in omega],
        axis=-1)

    mf._numint.block_loop = block_loop_decorator(mf._numint.block_loop)

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
        if hasattr(mf, 'U'):
            inp['U'] = mf.U[..., mf.mask_indices, :]
        parsed_xc = parse_xc_code(xc_code)
        total_output = {'v' + key: 0.0 for key in inp}
        total_output['zk'] = 0

        for code, factor in parsed_xc[1]:
            model = LibNXCFunctional(code, kind='grid')
            output = model.compute(inp)
            for key in output:
                if output[key] is not None:
                    total_output[key] += output[key] * factor

        exc, vlapl, vtau, vrho, vsigma, vU = [total_output.get(key,None)\
          for key in ['zk','vlapl','vtau','vrho','vsigma','vU']]

        mf.vU[..., mf.mask_indices,:] = vU
        vxc = (vrho, vsigma, vlapl, vtau)

        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative
        return exc, vxc, fxc, kxc

    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mf.U =  contract('mnQ, QP, Pki, ...mn-> ...ki',
            df_3c, df_2c_inv, vh_on_grid, dm)
        dft.libxc.define_xc_(mf._numint, eval_xc, max_level, hyb=hyb)
        mf.vU = np.zeros_like(mf.U)

        if method == dft.RKS:
            veff = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        else:
            veff = dft.uks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        vU = contract('mnQ, QP, Pki, ...ki,k-> ...mn',
            df_3c, df_2c_inv, vh_on_grid, mf.vU, mf.grids.weights)
        veff[:,:] += vU[:,:]
        return veff
    return get_veff
