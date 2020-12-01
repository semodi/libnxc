#include "defaults.h"
void Defaults::setDefault(const int code){
    switch(code){
        case SIESTA_ATOMIC_CODE:
          edens=0;
          add=1;
          gamma=1;
          break;
        case SIESTA_GRID_CODE:
          edens = 1;
          add = 0;
          cuda = 0;
          gamma = 1;
    }
}
void Defaults::useCuda(){
  cuda=1;
}
Defaults *Defaults::instance = 0;
Defaults *defaults = defaults->getInstance();
