#ifndef DEFAULTS_H
#define DEFAULTS_H

const int DEFAULT_CODE=0;
const int SIESTA_GRID_CODE=1;
const int SIESTA_ATOMIC_CODE=2;
const int CP2K_CODE=0;

class Defaults {
  static Defaults *instance;
  Defaults(){};
  public:
    int edens, add, cuda, gamma;
    static Defaults *getInstance() {
      if (!instance)
        instance = new Defaults;
      instance->edens = 1;
      instance->add = 0;
      instance->cuda = 0;
      instance->gamma = 0;
      return instance;
    }
    void setDefault(const int code);
    void useCuda();
  };
extern Defaults *defaults;

#endif
