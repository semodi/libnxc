#include "util.h"
#include "nxc_c.h"

#define XC_LDA_X_NUEG          821 /* Reference UEG exchange               */
#define XC_LDA_C_NPW92         822 /* Reference PW92 correlation           */

static void
lda_nxc_init(xc_func_type *p)
{
  if(nxc_c_xc_init(p->info->number, p->nspin)){
    fprintf(stderr, "Internal error in libnxc functional\n");
    exit(1);
  }
}

const xc_func_info_type xc_func_info_lda_x_nueg = {
  XC_LDA_X_NUEG,
  XC_EXCHANGE,
  "Reference UEG exchange",
  XC_FAMILY_LDA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  lda_nxc_init, NULL,
  nxc_c_lda_exc_vxc, NULL, NULL
};

const xc_func_info_type xc_func_info_lda_c_npw92 = {
  XC_LDA_C_NPW92,
  XC_CORRELATION,
  "Reference PW92 correlation",
  XC_FAMILY_LDA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  lda_nxc_init, NULL,
  nxc_c_lda_exc_vxc, NULL, NULL
};
