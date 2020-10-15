      PROGRAM test


      integer    :: na = 1
      integer    :: nua = 1
      integer   :: indxua(1)
      real(8)   :: xa(3,1), ucell(3,3)
      integer   :: grid(3)
      integer   :: isa(1) ! Species index for each atom
      real(8)   :: dvol


      character(len=2)      :: nxc_symbols(1)
      integer               :: ispec, ierr
      character(len=100)    :: nxc_path
      character(len=8)      :: GREEN= "\033[92m";
      character(len=8)      :: END_= "\033[0m";
      integer               :: MPIRank
      integer               :: myBox(2,3)
      real(8)              :: xua(3,1)
      integer               :: isua(1) ! Species index for each atom in uc
      integer               :: edens=0

      ! Initialize grid, basis, etc.
      nxc_path='test.jit'
      call nxc_f90_func_init(xua ,nua, ucell, grid, isua, nxc_symbols, &
        1, ierr, nxc_path, LEN_TRIM(nxc_path), myBox, edens)

      write(*,*) achar(27),'[92m',"OK!", &
         achar(27),'[0m'

      END PROGRAM test 

