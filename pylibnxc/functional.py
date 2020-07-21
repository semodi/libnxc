"""
neuralxc.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""
import numpy as np
import os
from glob import glob
import torch
from abc import ABC, abstractmethod


def LibNXCFunctional(**kwargs):

    if 'path' in kwargs:
        return AtomicFunc(kwargs['path'])
    else:
        raise Exception('So far only atomic functionals (NeuralXC) supported,' + \
            ' please specify path')


class NXCFunctional(ABC):

    def __init__(self):
        pass

    def initialize(self):
        pass

    def compute(self):
        pass

class AtomicFunc(NXCFunctional):

    def __init__(self, path):
        model_paths = glob(path + '/*')
        self.basis_models = {}
        self.projector_models = {}
        self.energy_models = {}
        print('NeuralXC: Loading model from ' + path)

        for mp in model_paths:
            if 'basis' in os.path.basename(mp):
                self.basis_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
            if 'projector' in  os.path.basename(mp):
                self.projector_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
            if 'xc' in os.path.basename(mp):
                self.energy_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
        print('NeuralXC: Model successfully loaded')

    def initialize(self, **kwargs):
        """Parameters
        ------------------
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        """
        self.projector_kwargs = kwargs
        periodic = (kwargs['unitcell'].shape == (3,3)) #TODO: this is just a workaround
        self.unitcell = torch.from_numpy(kwargs['unitcell']).double()
        # self.unitcell_inv = torch.inverse(self.unitcell).detach().numpy()
        self.epsilon = torch.zeros([3,3]).double()
        self.epsilon.requires_grad = True
        if periodic:
            self.unitcell_we = torch.mm((torch.eye(3) + self.epsilon), self.unitcell)
        else:
            self.unitcell_we = self.unitcell
        self.grid = torch.from_numpy(kwargs['grid']).double()
        self.positions = torch.from_numpy(kwargs['positions']).double()
        self.positions_we = torch.mm(torch.eye(3) + self.epsilon, self.positions.T).T
        # self.positions = torch.mm(self.positions_scaled,self.unitcell)
        self.species = kwargs['species']
        if periodic:
            U = torch.einsum('ij,i->ij', self.unitcell, 1/self.grid)
            self.V_cell = torch.abs(torch.det(U))
            self.V_ucell = torch.abs(torch.det(self.unitcell)).detach().numpy()
            self.my_box = torch.zeros([3,2])
            self.my_box[:,1] = self.grid
        else:
            self.V_cell = self.grid
            self.V_ucell = 1
            self.my_box = torch.zeros([3,2])
            self.my_box[:,1] = 1

        with torch.jit.optimized_execution(should_optimize=True):
            self.compute_basis(False)

    def compute_basis(self, positions_grad=False):
        self.positions.requires_grad = positions_grad
        self.positions_we = torch.mm(torch.eye(3) + self.epsilon, self.positions.T).T
        if positions_grad:
            unitcell = self.unitcell_we
            positions = self.positions_we
        else:
            unitcell = self.unitcell
            positions = self.positions

        self.radials = []
        self.angulars = []
        for pos, spec in zip(positions, self.species):
            rad, ang = self.basis_models[spec](pos, unitcell, self.grid, self.my_box)
            self.radials.append(rad)
            self.angulars.append(ang)

    def compute(self, inp, do_exc=True, do_vxc=True, do_forces=False):

        if isinstance(inp, np.ndarray):
            inp = {"rho": np.asarray(inp, dtype=np.double)}
        elif isinstance(inp, np.ndarray):
            inp = {"rho": np.asarray(inp["rho"], dtype=np.double)}

        if do_forces and not do_vxc:
            raise Exception('Vxc needs to be evaluated to compute pulay force correction')
        rho = inp["rho"]

        output = {}
        if do_forces:
            unitcell = self.unitcell_we
            positions = self.positions_we
        else:
            unitcell = self.unitcell
            positions = self.positions

        with torch.jit.optimized_execution(should_optimize=True):
            if do_forces:
                self.compute_basis(True)
            self.descriptors = {spec:[] for spec in self.species}
            rho = torch.from_numpy(rho).double()
            rho.requires_grad = True
            e_list = []
            for pos, spec, rad, ang in zip(positions, self.species,
                                                self.radials, self.angulars):
                e_list.append(self.energy_models[spec](
                    self.projector_models[spec](rho, pos,
                                                unitcell,
                                                self.grid,
                                                rad, ang, self.my_box).unsqueeze(0)
                                                )
                                            )

            E = torch.sum(torch.cat(e_list))
            output['zk'] = E.detach().numpy()
            if do_vxc:
                E.backward()
                V = (rho.grad/self.V_cell).detach().numpy()
                output['vrho'] = V
                if do_forces:
                    forces =  np.concatenate([-self.positions.grad.detach().numpy(),
                        self.epsilon.grad.detach().numpy()/self.V_ucell])
                    output['forces'] = forces

            return output
