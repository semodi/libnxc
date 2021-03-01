from pylibnxc import LibNXCFunctional
import pytest
import numpy as np
import os

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
