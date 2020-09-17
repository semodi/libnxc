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

        if not os.path.exists(path):
            raise Exception('Model not found, please if path correct')

        self.basis_models = {}
        self.projector_models = {}
        self.energy_models = {}
        self.spec_agn = False
        self.no_sc = False
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
            if 'NO_SC' in os.path.basename(mp):
                self.no_sc = True
            if 'AGN' in os.path.basename(mp):
                self.spec_agn = True


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
        if self.spec_agn:
            self.species = ['X' for s in self.species]
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
            self._compute_basis(False)

    def _compute_basis(self, positions_grad=False):
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
        self.boxes = []
        for pos, spec in zip(positions, self.species):
            rad, ang, box= self.basis_models[spec](pos, unitcell, self.grid, self.my_box)
            self.radials.append(rad)
            self.angulars.append(ang)
            self.boxes.append(box)

    def _compute_from_descriptors(self, inp, do_exc=True, do_vxc=True, do_forces=False):

        rho = inp["c"]
        output = {}

        with torch.jit.optimized_execution(should_optimize=True):
            rho = {spec: torch.from_numpy(rho[spec]).double() for spec in rho}
            if do_vxc:
                for spec in rho:
                    rho[spec].requires_grad = True
            e_list = []
            for spec in rho:
                e_list.append(self.energy_models[spec](rho[spec]))

            E = torch.sum(torch.cat(e_list))
            if do_vxc:
                E.backward()
                V = {spec: rho[spec].grad.detach().numpy() for spec in rho}
                output['dEdC'] = V
            output['zk'] = E.detach().numpy()

            return output

    def compute(self, inp, do_exc=True, do_vxc=True, do_forces=False):

        if isinstance(inp, np.ndarray):
            inp = {"rho": np.asarray(inp, dtype=np.double)}

        if self.no_sc:
            do_vxc = False
            do_forces = False

        if do_forces and not do_vxc:
            raise Exception('Vxc needs to be evaluated to compute pulay force correction')
        if "rho" in inp and "c" in inp:
            raise Exception('Error: Both density "rho" and descriptors "c" provided in input.')


        if "c" in inp:
            do_forces = False
            return self._compute_from_descriptors(inp, do_exc, do_vxc, do_forces)
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
                self._compute_basis(True)
            self.descriptors = {spec:[] for spec in self.species}
            rho = torch.from_numpy(rho).double()
            if do_vxc:
                rho.requires_grad = True
            e_list = []
            for pos, spec, rad, ang, box in zip(positions, self.species,
                                                self.radials, self.angulars,
                                                self.boxes):
                e_list.append(self.energy_models[spec](
                    self.projector_models[spec](rho, pos,
                                                unitcell,
                                                self.grid,
                                                rad, ang, box).unsqueeze(0)
                                                )
                                            )

            E = torch.sum(torch.cat(e_list))
            if do_vxc:
                E.backward()
                V = (rho.grad/self.V_cell).detach().numpy()
                output['vrho'] = V
                if do_forces:
                    forces =  np.concatenate([-self.positions.grad.detach().numpy(),
                        self.epsilon.grad.detach().numpy()/self.V_ucell])
                    output['forces'] = forces
            output['zk'] = E.detach().numpy()

            return output
