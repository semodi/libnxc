Fortran API
===================

The nomenclature used here strictly follows that of the C++ interface with the addition of
``_f90`` in the function names. Function signatures reflect the fact that values cannot be returned to Fortran.
To Initialize functionals from fortran, two methods are available:

   - ``nxc_f90_func_init_`` to initialize grid based functionals (LDA, GGA, ...) for which
   only the model name/path has to be provided.

   - ``nxc_f90_atmfunc_init_`` to initialize NeuralXC functionals for which more information about the simulation box is required.
::

  void nxc_f90_set_code_(int * code);
  void nxc_f90_use_cuda_();
  void nxc_f90_cuda_available(int * available);
  int nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
              char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr);
  int nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr);
  int nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr);
  int nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                              double forces[], double stress[], int* ierr);
  int nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
      double vrho[], double vsigma[], int* ierr);
  int nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
     double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr);
  void nxc_f90_func_get_family(int * family);
  void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family);


The trailing underscore in the function names is required for linking purposes and
has to be **dropped** when calling the function from fortran.
