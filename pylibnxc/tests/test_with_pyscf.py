import pylibnxc
import pytest
import numpy as np
import os
try:
    import pyscf
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_parse():
    parsed = pylibnxc.pyscf.utils.parse_xc_code('GGA_X_PBEd')
    assert parsed == ([0, 0, 0], [('GGA_X_PBED', 1)])
    parsed = pylibnxc.pyscf.utils.parse_xc_code('GGA_C_PBEd')
    assert parsed == ([0, 0, 0], [('GGA_C_PBED', 1)])
    parsed = pylibnxc.pyscf.utils.parse_xc_code('GGA_X_PBEd, GGA_C_PBEd')
    assert parsed == ([0, 0, 0], [('GGA_X_PBED', 1), ('GGA_C_PBED', 1)])
    parsed = pylibnxc.pyscf.utils.parse_xc_code(
        '0.5*GGA_X_PBEd + 0.75*MGGA_X_PBExx, GGA_C_PBEd + 0.5*HF')
    assert parsed == ([0.5, 0, 0], [('GGA_X_PBED', 0.5),
                                    ('MGGA_X_PBEXX', 0.75), ('GGA_C_PBED', 1)])
    assert pylibnxc.pyscf.utils.find_max_level(parsed) == 'MGGA'


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_pyscf():
    """ Simply test whether pyscf is working and giving the correct energies
    """
    from pyscf import gto, dft
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 5
    mf.kernel()
    assert all([np.allclose(this,ref) for this, ref in\
     zip(np.load(test_dir + '/ref/pyscf_h2o.npy'), [mf.e_tot, mf.get_veff().exc])])


def test_load_atomic():
    func = pylibnxc.LibNXCFunctional(name=test_dir +
                                     '/../../test/test.rad.jit',
                                     kind='atomic')
    assert type(func) == pylibnxc.functional.AtomicFunc


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_atomic():
    from pyscf import gto, dft
    nxc_path = test_dir + '/../../test/test.rad.jit'
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = pylibnxc.pyscf.RKS(mol, nxc=nxc_path, nxc_kind='atomic')
    mf.xc = 'PBE'
    mf.grids.level = 3
    mf.kernel()
    assert all([np.allclose(this,ref) for this, ref in\
     zip(np.load(test_dir + '/ref/h2o_atomic.npy'), [mf.e_tot])])


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('polarized', [False, True])
@pytest.mark.parametrize('func', ["LDA", "GGA", "MGGA"])
def test_hm(polarized, func):
    from pyscf import gto, dft
    nxc_path = func + '_HM'
    reference_values = {
        'LDA':
        (-7.29179794380983, -75.83484819708484),  # These ref. values were
        'GGA':
        (-7.42384149356207,
         -76.1038744376732),  # obtained with the original implementation
        'MGGA': (-7.46823966797756, -76.1998027485784)
    }[func]  # by Nagai et al.

    mol = gto.Mole()
    if polarized:
        mol.atom = """ 3 0 0 0"""
        mol.spin = 1
    else:
        mol.atom = """8            .000000     .000000     .119262
        1            .000000     .763239    -.477047
        1            .000000    -.763239    -.477047"""
        mol.spin = 0
    mol.charge = 0
    mol.basis = "6-31G"
    mol.build()
    if polarized:
        mf = pylibnxc.pyscf.UKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, reference_values[0])
    else:
        mf = pylibnxc.pyscf.RKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, reference_values[1])


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('name', ['H2', 'LiF', 'NO'])
@pytest.mark.parametrize('funcname', ['PBE_X', 'PBE'])
def test_nn_pbe(name, funcname):
    from pyscf import gto, dft
    # nxc_path = 'PBE_GGA'
    func = {
        'PBE_X': ['GGA_X_PBE', 'GGA_X_PBE'],
        'PBE': ['GGA_PBE', 'PBE'],
        'PBE_comp': ['GGA_X_PBE, GGA_C_PBE', 'PBE'],
        'PBE0': ['0.75*GGA_X_PBE+0.25*HF,GGA_C_PBE', 'PBE0'],
        'SCAN': ['MGGA_SCAN2', 'SCAN'],
        'revTPSS_X': ['MGGA_X_REVTPSS','MGGA_X_REVTPSS'],
        'revTPSS': ['MGGA_XC_REVTPSS','REVTPSS'],
    }[funcname]
    nxc_path = func[0]

    a_str = {
        'H2':
        """ 1            .000000    .000   -.377
                 1            .000000    .000    .377""",
        'LiF':
        """ 3            .000000    .000   -.70
                 9            .000000    .000    .70""",
        'NO':
        """ 8            .000000    .000   .000
                 7            .000000    .000    1.37""",
        'F2':
        """ 9   0   0   0
            9   0   0   1.42""",
        'CO2':
        """ 8            .000000    .000   -1.16
            8            .000000    .000   1.16
            6            .000000    .000    .000""",
        'N2C2':
        """ 7       0   0   1.8459
            7       0   0   -1.8459
            6       0   0   0.687868
            6       0   0   -0.687868"""
    }

    # if name == 'NO' and funcname == 'PBE':
    # pytest.xfail("PBE (NN) correlation not reliable in spin-polarized case")

    mol = gto.Mole()
    mol.atom = a_str[name]
    mol.spin = 0
    if name == 'NO':
        mol.spin = 1
    mol.charge = 0
    mol.verbose=4
    mol.basis = "6-311+G*"
    mol.build()
    results = []
    methods = [pylibnxc.pyscf.RKS, pylibnxc.pyscf.UKS]
    if mol.spin != 0:
        methods = methods[-1:]
    for method in methods:
        mf = method(mol, nxc=nxc_path)
        mf.grids.level = 5
        mf.kernel()
        assert mf.converged
        results.append(mf.e_tot)
    assert all([np.allclose(r, results[-1]) for r in results])
    nn_etot = results[0]

    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = func[1]
    mf.grids.level = 5
    mf.kernel()
    # assert mf.converged
    pbe_etot = mf.e_tot
    assert np.allclose(pbe_etot, nn_etot, atol=1e-4)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('name', ['H2', 'LiF', 'NO'])
@pytest.mark.parametrize('funcname', ['PBE_comp', 'PBE0'])
def test_nn_pbe_composite(name, funcname):
    test_nn_pbe(name, funcname)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('name', ['H2', 'LiF', 'NO','F2','CO2','N2C2'])
# @pytest.mark.parametrize('name', ['LiF'])
@pytest.mark.parametrize('funcname', ['SCAN'])
def test_nn_scan(name, funcname):
    test_nn_pbe(name, funcname)

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('name', ['H2', 'LiF', 'NO','F2','CO2','N2C2'])
# @pytest.mark.parametrize('name', ['LiF'])
# @pytest.mark.parametrize('funcname', ['revTPSS_X','revTPSS'])
@pytest.mark.parametrize('funcname', ['revTPSS'])
def test_nn_revtpss(name, funcname):
    test_nn_pbe(name, funcname)
