#include "util.h"
#include "nxc_c.h"
// #include "nxc_c_gga.h"

#define XC_MGGA_X_NSCAN          811 /* Machine learned SCAN X functional                */
#define XC_MGGA_C_NSCAN          812 /* Machine learned SCAN C functional                */
#define XC_MGGA_XC_NSCAN         813 /* Machine learned SCAN XC functional                */
#define XC_MGGA_XC_HM            814 /* Machine learned "Hidden messages" MGGA XC functional */
#define XC_MGGA_XC_MCUSTOM        815 /* Machine learned custom MGGA XC functional */

static void
mgga_nxc_init(xc_func_type *p)
{
  nxc_c_xc_init(p->info->number, p->nspin);
}


const xc_func_info_type xc_func_info_mgga_x_nscan = {
  XC_MGGA_X_NSCAN,
  XC_EXCHANGE,
  "Neural network MGGA exchange-correlation",
  XC_FAMILY_MGGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  mgga_nxc_init, NULL,
  NULL, NULL, nxc_c_mgga_exc_vxc
};

const xc_func_info_type xc_func_info_mgga_c_nscan = {
  XC_MGGA_C_NSCAN,
  XC_CORRELATION,
  "Neural network MGGA exchange-correlation",
  XC_FAMILY_MGGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  mgga_nxc_init, NULL,
  NULL, NULL, nxc_c_mgga_exc_vxc
};

const xc_func_info_type xc_func_info_mgga_xc_nscan = {
  XC_MGGA_XC_NSCAN,
  XC_EXCHANGE_CORRELATION,
  "Neural network MGGA exchange-correlation",
  XC_FAMILY_MGGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  mgga_nxc_init, NULL,
  NULL, NULL, nxc_c_mgga_exc_vxc
};

const xc_func_info_type xc_func_info_mgga_xc_hm = {
  XC_MGGA_XC_HM,
  XC_EXCHANGE_CORRELATION,
  "Neural network MGGA exchange-correlation",
  XC_FAMILY_MGGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  mgga_nxc_init, NULL,
  NULL, NULL, nxc_c_mgga_exc_vxc
};

const xc_func_info_type xc_func_info_mgga_xc_mcustom = {
  XC_MGGA_XC_MCUSTOM,
  XC_EXCHANGE_CORRELATION,
  "Neural network custom MGGA exchange-correlation",
  XC_FAMILY_MGGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  mgga_nxc_init, NULL,
  NULL, NULL, nxc_c_mgga_exc_vxc
};
