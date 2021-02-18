Libnxc Documentation
=========================================================


Libnxc is a libary to use **machine learned** exchange-correlation functionals for density functional theory.
All common functional types (LDA, GGA, metaGGA) as well as NeuralXC type functionals  are supported.
Libnxc is written in C++ and has Fortran bindings. An implementation in Python, `pylibnxc` is also available.
Libnxc is inspired by Libxc, mirroring as closely  as possible its API. In doing so, the integration of Libnxc in electronic structure codes that use Libxc should be straightforward.
Libnxc can utilize multi-processing through MPI and model inference on GPUs through CUDA is supported as well.
Although the primary motivation for Libnxc was to add support for neural network based functionals, other types of models can be used as well.
As long as the following requirements are fulfilled, models can be used by Libnxc:

  1. The model has to be implemented in PyTorch and serialized into a TorchScript model (e.g. with ``torch.jit.trace``)
  2. The model input and output has to follow the form specified in :ref:`Functionals`

The serialized model is regarded as a containerized black box by Libnxc.
Thus, even simple polynomial models can be implemented and evaluated.
While not replacing hard-coded functionals such as the ones employed by Libxc and directly by DFT codes,
this approach provides several advantages:

  - **Fast experimentation**: Functionals can be quickly implemented and used in a `plug-and-play` manner
  - **Automatic differentiation**: PyTorch takes care of calculating all derivative terms needed in the exchange-correlation potential.
  - **Native GPU support**: PyTorch is designed to be run on GPUs using CUDA. This extends to serialized TorchScript models, thus
    evaluating libnxc functionals on GPUs is straightforward.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   quickstart.rst
   functionals.rst
   pyscf.rst
   libxc.rst
   cpp.rst
   fortran.rst
   pylibnxc.rst
