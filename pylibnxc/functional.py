"""
functional.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""
import numpy as np
import os
from glob import glob
import torch
from abc import ABC, abstractmethod

_default_modelpath = os.path.dirname(__file__) + '/../models/'


def LibNXCFunctional(name, **kwargs):
    """ Loads a Libnxc functional
    Parameters
    ----------
    name, string
        Either the name of a pre-defined functional or path to custom
        functional (will check path first and then resort to pre-defined functional)
    kind, string, optional {'grid', 'atomic'}
        default: 'grid', whether functional is grid kind (LDA, GGA etc.) or
        atomic (NeuralXC)

    Returns
    --------
    NXCFunctional
    """
    #Resolve path
    model_path = os.environ.get('NXC_MODELPATH', _default_modelpath)

    #Check if name is path
    if os.path.exists(name):
        path = name
    elif os.path.exists(os.path.join(model_path, name)):
        path = os.path.join(model_path, name)
    else:
        raise ValueError(
            'Model {} could not be found, please check name/path'.format(name))

    if kwargs.get('kind', 'grid').lower() == 'grid':
        if 'HM' in name:
            func = HMFunc(path)
        else:
            func = GridFunc(path)
        func._xctype = os.path.basename(name).split('_')[0]
        return func
    elif kwargs.get('kind', 'grid').lower() == 'atomic':
        return AtomicFunc(name)


class NXCFunctional(ABC):
    def __init__(self, path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise Exception(
                'Model not found at {}, please check if path correct'.format(
                    path))
        self.energy_model =\
                 torch.jit.load(path + '/xc')

        if os.path.exists(path + '/omega'):
            self.omega = np.genfromtxt(path +'/omega').tolist()
        else:
            self.omega = []
            
        self.safe_mode = True

    def initialize(self):
        pass

    def compute(self):
        pass


class GridFunc(NXCFunctional):

    # _gamma_eps = 1e-8
    _gamma_eps = 0

    def compute(self, inp, do_exc=True, do_vxc=True, **kwargs):
        """ Evaluate the functional on a given input

        Parameters
        ----------
        inp, dict of np.ndarrays
            Input electron density "rho" and its derivatives. Potential terms are
            calculated for all provided derivates.
        do_exc, bool, optional
            no effect, only here for compatibility reasons
        do_vxc, bool, optional
            whether to compute the functional derivative(s) of the energy.
            default: True

        Returns:
        ---------
        output, dict
            Dictionary containing output values:
                - 'zk': energy per unit particle or total energy
                - 'vrho/vsigma/vtau' : potential terms
        """

        spin = (inp['rho'].ndim == 2)
        inputs = []
        rho0 = torch.from_numpy(inp['rho'])
        inputs.append(rho0)
        if 'sigma' in inp:
            drho = torch.from_numpy(inp['sigma'])
            inputs.append(drho)
        elif 'gamma' in inp:
            drho = torch.from_numpy(inp['gamma'])
            drho.requires_grad = True
            if spin:
                dxa, dya, dza = drho[0, :]
                dxb, dyb, dzb = drho[1, :]
            else:
                dxa, dya, dza = drho * 0.5
                dxb, dyb, dzb = drho * 0.5

            sigma_a = (dxa**2 + dya**2 + dza**2) + self._gamma_eps
            sigma_b = (dxb**2 + dyb**2 + dzb**2) + self._gamma_eps
            sigma_ab = (dxb * dxa + dyb * dya + dzb * dza) + self._gamma_eps
            inputs.append(drho)
        if 'tau' in inp:
            tau = torch.from_numpy(inp['tau'])
            inputs.append(tau)
        if 'U' in inp:
            U = torch.from_numpy(inp['U'])
            inputs.append(U)

        if do_vxc:
            for idx, _ in enumerate(inputs):
                inputs[idx].requires_grad = True

        torch_inputs = []
        if spin:
            rho0_a = rho0[0]
            rho0_b = rho0[1]
            if 'sigma' in inp and not 'gamma' in inp:
                sigma_a, sigma_ab, sigma_b = drho + self._gamma_eps
            if 'tau' in inp:
                tau_a, tau_b = tau
            if 'U' in inp:
                nl_a, nl_b = U


        else:
            rho0_a = rho0_b = rho0 * 0.5
            if 'sigma' in inp and not 'gamma' in inp:
                sigma_a = sigma_b = sigma_ab = drho * 0.25 + self._gamma_eps
            if 'tau' in inp:
                tau_a = tau_b = tau * 0.5
            if 'U' in inp:
                nl_a = nl_b = U*0.5

        torch_inputs.append(rho0_a.unsqueeze(-1))
        torch_inputs.append(rho0_b.unsqueeze(-1))
        if 'gamma' in inp or 'sigma' in inp:
            torch_inputs.append(sigma_a.unsqueeze(-1))
            torch_inputs.append(sigma_ab.unsqueeze(-1))
            torch_inputs.append(sigma_b.unsqueeze(-1))
        if 'tau' in inp:
            torch_inputs.append(torch.zeros_like(
                tau_a.unsqueeze(-1)))  # Expects laplacian in input
            torch_inputs.append(torch.zeros_like(
                tau_b.unsqueeze(-1)))  # even though not used
            torch_inputs.append(tau_a.unsqueeze(-1))
            torch_inputs.append(tau_b.unsqueeze(-1))
        if 'U' in inp:
            torch_inputs.append(nl_a)
            torch_inputs.append(nl_b)

        torch_inputs = torch.cat(torch_inputs, dim=-1)
        exc = self.energy_model(torch_inputs)[:, 0]
        assert exc.dim() == 1
        E = torch.dot(exc, torch_inputs[:, 0] + torch_inputs[:, 1])

        if do_vxc:
            E.backward()
        exc = exc.detach().numpy()

        outputs = {}
        outputs['zk'] = exc

        if do_vxc:
            outputs['vrho'] = rho0.grad.detach().numpy().T
            if 'sigma' in inp:
                outputs['vsigma'] = drho.grad.detach().numpy().T
            elif 'gamma' in inp:
                outputs['vgamma'] = drho.grad.detach().numpy()
            if 'tau' in inp:
                outputs['vtau'] = tau.grad.detach().numpy().T
            if 'U' in inp:
                outputs['vU'] = U.grad.detach().numpy()

        if self.safe_mode:
            for key in outputs:
                outputs[key] = np.nan_to_num(outputs[key], copy=False)
        return outputs


class HMFunc(GridFunc):
    _gamma_eps = 1e-7


class AtomicFunc(NXCFunctional):
    def __init__(self, path):
        path = os.path.abspath(path)
        model_paths = glob(path + '/*')
        if not os.path.exists(path):
            raise Exception(
                'Model not found at {}, please check if path correct'.format(
                    path))

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
            if 'projector' in os.path.basename(mp):
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
        periodic = 'unitcell' in kwargs
        if periodic:
            self.unitcell = torch.from_numpy(kwargs['unitcell']).double()
        else:
            self.unitcell = torch.from_numpy(kwargs['grid_coords']).double()
        # self.unitcell_inv = torch.inverse(self.unitcell).detach().numpy()
        self.epsilon = torch.zeros([3, 3]).double()
        self.epsilon.requires_grad = True
        self.periodic = periodic
        if periodic:
            self.unitcell_we = torch.mm((torch.eye(3) + self.epsilon),
                                        self.unitcell)
        else:
            self.unitcell_we = self.unitcell
        if periodic:
            self.grid = torch.from_numpy(kwargs['grid']).double()
        else:
            self.grid = torch.from_numpy(kwargs['grid_weights']).double()

        self.positions = torch.from_numpy(kwargs['positions']).double()
        self.positions_we = torch.mm(
            torch.eye(3) + self.epsilon, self.positions.T).T
        # self.positions = torch.mm(self.positions_scaled,self.unitcell)
        self.species = kwargs['species']
        if self.spec_agn:
            self.species = ['X' for s in self.species]
        if periodic:
            U = torch.einsum('ij,i->ij', self.unitcell, 1 / self.grid)
            self.V_cell = torch.abs(torch.det(U))
            self.V_ucell = torch.abs(torch.det(self.unitcell)).detach().numpy()
            self.my_box = torch.zeros([3, 2])
            self.my_box[:, 1] = self.grid
        else:
            self.V_cell = self.grid
            self.V_ucell = 1
            self.my_box = torch.zeros([3, 2])
            self.my_box[:, 1] = 1

        with torch.jit.optimized_execution(should_optimize=True):
            self._compute_basis(False)

    def _compute_basis(self, positions_grad=False):
        self.positions.requires_grad = positions_grad
        self.positions_we = torch.mm(
            torch.eye(3) + self.epsilon, self.positions.T).T
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
            rad, ang, box = self.basis_models[spec](pos, unitcell, self.grid,
                                                    self.my_box)
            self.radials.append(rad)
            self.angulars.append(ang)
            self.boxes.append(box)

    def _compute_from_descriptors(self,
                                  inp,
                                  do_exc=True,
                                  do_vxc=True,
                                  do_forces=False):

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

    def compute(self,
                inp,
                do_exc=True,
                do_vxc=True,
                do_forces=False,
                edens=True):
        """ Evaluate the functional on a given input

        Parameters
        ----------
        inp, dict of np.ndarrays
            Input electron density "rho" or ML descriptors "c". If "c" is provided
            projecting the electron density is skipped and _compute_from_descriptors
            is called directly.
        do_exc, bool, optional
            no effect, only here for compatibility reasons
        do_vxc, bool, optional
            whether to compute the functional derivative(s) of the energy.
            default: True
        do_forces, bool, optional
            whether to compute pulay corrections to forces. Only possible
            if electron density and not descriptors provided. default: False
        edens: bool, optional
            whether to return the energy per unit particle or the total energy,
            default: True

        Returns:
        ---------
        output, dict
            Dictionary containing output values:
                - 'zk': energy per unit particle or total energy
                - 'vrho/vsigma/vtau' : potential terms
                - 'forces': force corrections
        """

        if isinstance(inp, np.ndarray):
            inp = {"rho": np.asarray(inp, dtype=np.double)}

        if self.no_sc:
            do_vxc = False
            do_forces = False

        if do_forces and not do_vxc:
            raise Exception(
                'Vxc needs to be evaluated to compute pulay force correction')
        if "rho" in inp and "c" in inp:
            raise Exception(
                'Error: Both density "rho" and descriptors "c" provided in input.'
            )

        if "c" in inp:
            do_forces = False
            return self._compute_from_descriptors(inp, do_exc, do_vxc,
                                                  do_forces)
        rho = inp["rho"]

        output = {}
        if do_forces:
            unitcell = self.unitcell_we
            positions = self.positions_we
        else:
            unitcell = self.unitcell
            positions = self.positions

        rho_np = rho
        with torch.jit.optimized_execution(should_optimize=True):
            if do_forces:
                self._compute_basis(True)
            self.descriptors = {spec: [] for spec in self.species}
            rho = torch.from_numpy(rho).double()
            if do_vxc:
                rho.requires_grad = True
            e_list = []
            for pos, spec, rad, ang, box in zip(positions, self.species,
                                                self.radials, self.angulars,
                                                self.boxes):
                e_list.append(self.energy_models[spec](
                    self.projector_models[spec](rho, pos, unitcell, self.grid,
                                                rad, ang, box).unsqueeze(0)))

            E = torch.sum(torch.cat(e_list))
            if do_vxc:
                E.backward()
                V = (rho.grad / self.V_cell).detach().numpy()
                output['vrho'] = V
                if do_forces:
                    forces = np.concatenate([
                        -self.positions.grad.detach().numpy(),
                        self.epsilon.grad.detach().numpy() / self.V_ucell
                    ])
                    output['forces'] = forces

            E = E.detach()
            if edens:
                qtot = torch.sum(rho * self.V_cell).detach().numpy()
                if self.periodic:
                    grid_factor = len(rho.flatten()) / torch.prod(
                        self.grid).detach().numpy()
                    output['zk'] = E / qtot * np.ones_like(
                        rho_np) * grid_factor
                else:
                    output['zk'] = E / self.grid / qtot / len(self.grid)
            else:
                output['zk'] = E

            return output
