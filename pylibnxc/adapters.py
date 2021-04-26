from abc import ABC, abstractmethod

import numpy as np

from .functional import LibNXCFunctional

Hartree = 27.211386024367243


def get_nxc_adapter(kind, path, options={}):
    """ Adapter factory for NeuralXC
    """
    kind = kind.lower()
    adapter_dict = {'pyscf': PySCFNXC}
    if not kind in adapter_dict:
        raise ValueError('Selected Adapter not available')
    else:
        adapter = adapter_dict[kind](path, options)
    return adapter


class NXCAdapter(ABC):
    def __init__(self, path, *args):
        path = ''.join(path.split())
        self._adaptee = LibNXCFunctional(name=path, kind='atomic')
        self.initialized = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def compute(self):
        pass


class PySCFNXC(NXCAdapter):
    def initialize(self, grid_coords, grid_weights, mol):
        """ PySCF adapter for atomic (NeuralXC) models
        """
        self.initialized = True
        self.grid_weights = np.array(grid_weights)
        self._adaptee.initialize(
            grid_coords=np.array(grid_coords),
            grid_weights=np.array(grid_weights),
            positions=mol.atom_coords(),
            species=[mol.atom_symbol(i) for i in range(mol.natm)])

    def compute(self, *args, **kwargs):
        output = self._adaptee.compute(*args, **kwargs)
        E = output.get('zk', 0)
        V = output.get('vrho', 0)
        E /= Hartree
        V /= Hartree
        return E, V
