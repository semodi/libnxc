#include "nxc_func.h"
std::shared_ptr<NXCFunc> get_functional(std::string model)
{

  bool found = false;

  std::for_each(funcs.begin(),funcs.end(),
                [&model, &found](const std::pair<int,std::string> &p){
                  if (p.second == model)
                    found = true;
                });

  if (found)
      switch(xctypes.at(model)){
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

  return std::make_shared<AtomicFunc>(model);
}
