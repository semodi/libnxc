#include "util.h"
#include "nxc_c.h"
// #include "nxc_c_gga.h"

#define XC_GGA_X_NPBE          801 /* Machine learned PBE X functional                */
#define XC_GGA_C_NPBE          802 /* Machine learned PBE C functional                */
#define XC_GGA_XC_NPBE         803 /* Machine learned PBE XC functional               */
#define XC_GGA_XC_HM           804 /* Machine learned "Hidden messages" GGA XC functional  */
#define XC_GGA_XC_KSR          805 /* Machine learned "Kohn Sham regularized" XC functional */
#define XC_GGA_XC_CUSTOM       806 /* Machine learned custom functional */

static void
gga_nxc_init(xc_func_type *p)
{
  nxc_c_xc_init(p->info->number);
}

const xc_func_info_type xc_func_info_gga_xc_npbe = {
  XC_GGA_XC_NPBE,
  XC_EXCHANGE_CORRELATION,
  "Neural network PBE exchange-correlation",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};

const xc_func_info_type xc_func_info_gga_x_npbe = {
  XC_GGA_X_NPBE,
  XC_EXCHANGE,
  "Neural network PBE exchange",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};

const xc_func_info_type xc_func_info_gga_c_npbe = {
  XC_GGA_C_NPBE,
  XC_CORRELATION,
  "Neural network PBE correlation",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};

const xc_func_info_type xc_func_info_gga_xc_hm = {
  XC_GGA_XC_HM,
  XC_EXCHANGE_CORRELATION,
  "Hidden messages GGA exchange-correlation",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};

const xc_func_info_type xc_func_info_gga_xc_ksr = {
  XC_GGA_XC_KSR,
  XC_EXCHANGE_CORRELATION,
  "KSR GGA exchange-correlation",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};

const xc_func_info_type xc_func_info_gga_xc_custom = {
  XC_GGA_XC_CUSTOM,
  XC_EXCHANGE_CORRELATION,
  "Neural network custom exchange-correlation",
  XC_FAMILY_GGA,
  {NULL, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-32,
  0, NULL, NULL,
  gga_nxc_init, NULL,
  NULL, nxc_c_gga_exc_vxc, NULL
};
