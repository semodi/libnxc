Functionals
=================

Anatomy of a Libnxc functional
-------------------------------

Libnxc functionals are provided as serialized TorchScript models. The binary file containing the
serialzed model should be called `xc` and reside within a directory whose name corresponds to the
name of the functional.

| Example
| ├── GGA_X_PBE
| │   ├── xc
| ├── GGA_C_PBE
| │   ├── xc

corresponds to two functionals, PBE exchange and PBE correlation. We followed the nomenclature of Libxc ``FAMILY_TYPE_NAME``, however
Libnxc only requires that the family (Jacob's ladder rung) is specified, the type (exchange or correlation or both) can be ommited.

The model expects **exactly one variable as input**: A two dimensional tensor containing the density and its derivatives in the following order
:math:`(\rho_\alpha, \rho_\beta, \gamma_\alpha, \gamma_{\alpha\beta}, \gamma_{\beta}, \nabla^2\rho_{\alpha}, \nabla^2\rho_{\beta}, \tau_\alpha, \tau_\beta)`,
where :math:`\alpha` and :math:`\beta` correspond to the spin up and down channels.
The shape of the tensor is therefore :math:`(N_{grid}, 9)`. For GGA type functionals, :math:`\nabla^2\rho` and  :math:`\tau` can dropped.

The model should return the **energy per unit particle** :math:`\epsilon_{xc}` as a 1-dimensional tensor of shape :math:`(N_{grid})`.

In order to take care of PyTorch's autodiff capabilities the model should be defined as a ``torch.nn.Module``.

This could look like the following simplified example:::

    class ExampleXC(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, rho):
        rho_a = rho[:, 0]
        rho_b = rho[:, 1]
        gamma_a = rho[:, 2]
        gamma_ab = rho[:, 3]
        gamma_b = rho[:, 4]
        lapl_a = rho[:, 5]
        lapl_b = rho[:, 6]
        tau_a = rho[:, 7]
        tau_b = rho[:, 8]

        exc = (rho_a + rho_b)**(4/3)
        return exc


The model can then be serialized with ::

  serialized = torch.jit.trace(forward, torch.abs(torch.rand(100,9)))
  torch.jit.save(serialzed, 'MGGA_XC_EXAMPLE/xc')

Shipped functionals
--------------------
