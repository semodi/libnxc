import os

import numpy as np

import pytest
from pylibnxc import LibNXCFunctional

try:
    import pyscf
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))


def test_pbe():
    func = LibNXCFunctional("GGA_XC_PBE")
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    inp = {'rho': rho, 'sigma': sigma}

    results = func.compute(inp)

    stacked = np.stack([inp['rho']] + [results[key] for key in results])
    # np.save('pbe_expected_output.npy',stacked)
    assert np.allclose(np.load(test_dir + '/pbe_expected_output.npy'), stacked)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_pbe_comp_pyscf():
    from pyscf import gto, dft

    func = LibNXCFunctional("GGA_X_PBE")
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    gamma = np.sqrt(sigma)
    gamma = np.stack(3 * [1 / np.sqrt(3) * gamma], axis=0)
    assert np.allclose(np.sum(gamma**2, axis=0), sigma)
    inp = {'rho': rho, 'gamma': gamma}

    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)

    results_pyscf = mf._numint.eval_xc(
        'GGA_X_PBE', np.concatenate([rho.reshape(1, -1), gamma]))
    results = func.compute(inp)
    assert np.allclose(results_pyscf[0], results['zk'], atol=1e-6)
    assert np.allclose(results_pyscf[1][0], results['vrho'], atol=1e-3)


def test_absolute_path():
    func = LibNXCFunctional(test_dir + "/../../models/GGA_XC_PBE")


def test_pbe_gamma():
    func = LibNXCFunctional("GGA_XC_PBE")
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    gamma = np.sqrt(sigma)
    gamma = np.stack(3 * [1 / np.sqrt(3) * gamma], axis=0)
    assert np.allclose(np.sum(gamma**2, axis=0), sigma)
    inp = {'rho': rho, 'gamma': gamma}

    results = func.compute(inp)
    vgamma = results['vgamma']

    vsigma = vgamma / (2 * gamma)
    results['vgamma'] = vsigma[0, :]
    stacked = np.stack([inp['rho']] + [results[key] for key in results])
    reference = np.load(test_dir + '/pbe_expected_output.npy')

    for idx, (st, ref) in enumerate(zip(stacked, reference)):
        assert np.allclose(st, ref)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_scan_comp_pyscf():
    from pyscf import gto, dft

    # def get_tau(rho, sigma, alpha):
    #     uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
    #     return (sigma/(8*rho))+(uniform_factor*rho**(5/3))*alpha
    #
    # def get_sigma(rho, s):
    #     return (s*2*(3*np.pi**2)**(1/3)*rho**(4/3))**2

    func = LibNXCFunctional("MGGA_X_SCAN")
    # rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # sigma = get_sigma(rho, 0.4)
    # lapl = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # tau = get_tau(rho, sigma, 1)
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    lapl = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    tau = np.array([0.3, 0.3, 0.4, 0.5, 0.6])
    gamma = np.sqrt(sigma)
    gamma = np.stack(3 * [1 / np.sqrt(3) * gamma], axis=0)
    assert np.allclose(np.sum(gamma**2, axis=0), sigma)
    inp = {'rho': rho, 'gamma': gamma, 'tau': tau}

    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)

    results_pyscf = mf._numint.eval_xc(
        'MGGA_X_SCAN',
        np.concatenate([
            rho.reshape(1, -1), gamma,
            lapl.reshape(1, -1),
            tau.reshape(1, -1)
        ]))
    results = func.compute(inp)
    assert np.allclose(results_pyscf[0], results['zk'], atol=1e-3)
    assert np.allclose(results_pyscf[1][0], results['vrho'], atol=1e-2)
    assert np.allclose(results_pyscf[1][3], results['vtau'], atol=1e-2)
