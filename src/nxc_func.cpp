#include "nxc_func.h"
std::shared_ptr<NXCFunc> get_functional(std::string model)
{

  int cnt = 0;
  for (const auto& name: model_names){
    if (model==name){
      switch(model_types[cnt]){
        case LDA_TYPE:
           return std::make_shared<LDAFunc>(model);
           break;
        case GGA_TYPE:
           return std::make_shared<GGAFunc>(model);
           break;
        case MGGA_TYPE:
           return std::make_shared<MGGAFunc>(model);
           break;
      }
    }
    ++cnt;
  }
  return std::make_shared<AtomicFunc>(model);
}
