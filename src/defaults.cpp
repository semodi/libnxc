#include "defaults.h"
void Defaults::setDefault(const int code){
    switch(code){
        case SIESTA_CODE:
          edens=0;
          add=1;
          break;
    }
}
void Defaults::useCuda(){
  cuda=1;
}
Defaults *Defaults::instance = 0;
Defaults *defaults = defaults->getInstance();
