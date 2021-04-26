.. _Functionals:

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


Unless a shipped functional (see below) is being used, models have to be loaded as **GGA_XC_CUSTOM** and **MGGA_XC_MCUSTOM**,
for GGA and meta-GGA functionals respectively. Using these labels will instruct Libnxc to search for models of the same name in
the current working directory. This is usually the directory from which the electronic structure calculation is run. To use a custom
functional, it should either be copied or linked to a model in the working directory with the name **GGA_XC_CUSTOM** or *MGGA_XC_MCUSTOM**.

Shipped functionals
--------------------

The following functionals were introduced in

[1] *Nagai, Ryo, Ryosuke Akashi, and Osamu Sugino. "Completing density functional theory by machine learning hidden messages from molecules." npj Computational Materials 6.1 (2020): 1-8.*

Please consider citing the paper when using them.

- **LDA_HM**: NN-LSDA introduced in [1]
- **GGA_HM**: NN-GGA introduced in [1]
- **MGGA_HM**: NN-meta-GGA introduced in [1]


The following functionals are only included for testing purposes. For small molecules an accuracy of about 1 mHartree can be expected.

- **GGA_XC_PBE**: Neural Network fitted to reproduce the popular PBE functional
- **GGA_X_PBE**: Exchange part of GGA_PBE
- **GGA_C_PBE**: Correlation part of GGA_PBE

- **MGGA_XC_SCAN**: Neural Network fitted to reproduce the popular PBE functional
- **MGGA_X_SCAN**: Exchange part of GGA_PBE
- **MGGA_C_SCAN**: Correlation part of GGA_PBE
