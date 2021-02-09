#include <map>

const int LDA_TYPE=0;
const int GGA_TYPE=1;
const int MGGA_TYPE=2;
const int ATOMIC_TYPE=4;

const std::map<int,std::string> funcs=
{
  {800, "LDA_HM"},
  {801, "GGA_X_PBE"},
  {802, "GGA_C_PBE"},
  {803, "GGA_PBE"},
  {804, "GGA_HM"},
  {805, "GGA_KSR"},
  {806, "GGA_XC_CUSTOM"},
  {810, "MGGA_X_PBE"},
  {811, "MGGA_X_SCAN"},
  {812, "MGGA_C_SCAN"},
  {813, "MGGA_SCAN"},
  {814, "MGGA_HM"},
  {815, "MGGA_XC_CUSTOM"}
};

const std::map<std::string,int> xctypes=
{
  {"LDA_HM", LDA_TYPE},
  {"GGA_X_PBE", GGA_TYPE},
  {"GGA_C_PBE", GGA_TYPE},
  {"GGA_PBE", GGA_TYPE},
  {"GGA_HM", GGA_TYPE},
  {"GGA_KSR", GGA_TYPE},
  {"GGA_XC_CUSTOM", GGA_TYPE},
  {"MGGA_X_PBE", MGGA_TYPE},
  {"MGGA_X_SCAN", MGGA_TYPE},
  {"MGGA_C_SCAN", MGGA_TYPE},
  {"MGGA_SCAN", MGGA_TYPE},
  {"MGGA_HM", MGGA_TYPE},
  {"MGGA_XC_CUSTOM", MGGA_TYPE}
};
