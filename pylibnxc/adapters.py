from abc import ABC, abstractmethod
from .functional import LibNXCFunctional
import numpy as np
Hartree = 27.211386024367243

def get_nxc_adapter(kind, path, options={}):
    """ Adapter factory for NeuralXC
    """
    kind = kind.lower()
    adapter_dict = { 'pyscf': PySCFNXC}
    if not kind in adapter_dict:
        raise ValueError('Selected Adapter not available')
    else:
        adapter = adapter_dict[kind](path, options)
    return adapter


class NXCAdapter(ABC):
    def __init__(self, path, *args):
        path = ''.join(path.split())
        self._adaptee = LibNXCFunctional(path=path)
        self.initialized = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def compute(self):
        pass

class PySCFNXC(NXCAdapter):

    def initialize(self, grid_coords, grid_weights, mol):
        self.initialized = True
        self.grid_weights = np.array(grid_weights)
        self._adaptee.initialize(unitcell=np.array(grid_coords), grid=np.array(grid_weights),
        positions=mol.atom_coords(), species=[mol.atom_symbol(i) for i in range(mol.natm)])

    def compute(self, *args, **kwargs):
        output = self._adaptee.compute(*args, **kwargs)
        E = output['zk']
        V = output['vrho']
        E /= Hartree
        V /= Hartree
        return E, V
