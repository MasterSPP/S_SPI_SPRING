#pragma once
#include "datatypes.h"
#include <cstdio>
#include <cmath>
#include <cmath>
#include <memory>
#include <cstring>
#include <omp.h>

#include "context.h"



class CPU_context: public Context{
public:
    CPU_context(int npixels_, int seed_,  FLOAT* intensities_): Context(npixels_, seed_)
        {
            
        
    }
    
    
    ~CPU_context(){


    }
    
    
};



