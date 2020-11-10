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
    func = pylibnxc.LibNXCFunctional(path = test_dir + '/../../test/test.rad.jit')
    assert type(func) == pylibnxc.functional.AtomicFunc

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_atomic():
    from pyscf import gto, dft
    nxc_path = test_dir + '/../../test/test.rad.jit'
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = pylibnxc.pyscf.RKS(mol, nxc = nxc_path)
    mf.xc = 'PBE'
    mf.grids.level = 3
    mf.kernel()
    assert all([np.allclose(this,ref) for this, ref in\
     zip(np.load(test_dir + '/ref/h2o_atomic.npy'), [mf.e_tot])])


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('polarized',[False,True])
def test_hm_lda(polarized):
    from pyscf import gto, dft
    nxc_path = 'HM_LDA'
    mol = gto.Mole()
    if polarized:
        mol.atom=""" 3 0 0 0"""
        mol.spin = 1
    else:
        mol.atom="""8            .000000     .000000     .119262
        1            .000000     .763239    -.477047
        1            .000000    -.763239    -.477047"""
        mol.spin  =0
    mol.charge=0
    mol.basis = "6-31G"
    mol.build()
    if polarized:
        mf = pylibnxc.pyscf.UKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot,  -7.29179794380983)
    else:
        mf = pylibnxc.pyscf.RKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, -75.83484819708484)

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('polarized',[False,True])
def test_hm_gga(polarized):
    from pyscf import gto, dft
    nxc_path = 'HM_GGA'
    mol = gto.Mole()
    if polarized:
        mol.atom=""" 3 0 0 0"""
        mol.spin = 1
    else:
        mol.atom="""8            .000000     .000000     .119262
        1            .000000     .763239    -.477047
        1            .000000    -.763239    -.477047"""
        mol.spin  =0
    mol.charge=0
    mol.verbose=4
    mol.basis = "6-31G"
    mol.build()
    if polarized:
        mf = pylibnxc.pyscf.UKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, -7.42384149356207)
    else:
        mf = pylibnxc.pyscf.RKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, -76.1038744376732)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('polarized',[False,True])
def test_hm_mgga(polarized):
    from pyscf import gto, dft
    nxc_path = 'HM_MGGA'
    mol = gto.Mole()
    if polarized:
        mol.atom=""" 3 0 0 0"""
        mol.spin = 1
    else:
        mol.atom="""8            .000000     .000000     .119262
        1            .000000     .763239    -.477047
        1            .000000    -.763239    -.477047"""
        mol.spin  =0
    mol.charge=0
    mol.basis = "6-31G"
    mol.build()
    if polarized:
        mf = pylibnxc.pyscf.UKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, -7.46823966797756)
    else:
        mf = pylibnxc.pyscf.RKS(mol, nxc=nxc_path)
        mf.kernel()
        assert np.allclose(mf.e_tot, -76.1998027485784)

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('name',['H2','LiF','NO'])
@pytest.mark.parametrize('funcname',['PBE_X', 'PBE'])
def test_nn_pbe(name, funcname):
    from pyscf import gto, dft
    # nxc_path = 'PBE_GGA'
    func = {'PBE_X':['PBE_X_GGA', 'GGA_X_PBE'],
            'PBE': ['PBE_GGA', 'PBE']}[funcname]
    nxc_path = func[0]

    a_str = {'H2': """ 1            .000000    .000   -.377
                 1            .000000    .000    .377""",
             'LiF': """ 3            .000000    .000   -.70
                 9            .000000    .000    .70""",
             'NO': """ 8            .000000    .000   .000
                 7            .000000    .000    1.37"""}
                 
    if name == 'NO' and funcname == 'PBE':
        pytest.xfail("PBE (NN) correlation not reliable in spin-polarized case")

    mol = gto.Mole()
    mol.atom=a_str[name]
    mol.spin  =0
    if name == 'NO':
        mol.spin = 3
    mol.charge=0
    mol.basis = "6-311+G*"
    mol.build()
    results = []
    methods = [pylibnxc.pyscf.RKS,
               pylibnxc.pyscf.UKS]
    if mol.spin != 0:
        methods =methods[-1:]
    for method in methods:
        mf = method(mol, nxc=nxc_path)
        mf.grids.level = 4
        mf.kernel()
        results.append(mf.e_tot)
    assert all([np.allclose(r, results[-1]) for r in results])
    nn_etot = results[0]

    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = func[1]
    mf.grids.level = 4
    mf.kernel()
    pbe_etot = mf.e_tot
    assert np.allclose(pbe_etot, nn_etot, atol=1e-3)
