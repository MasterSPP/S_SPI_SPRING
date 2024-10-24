#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <memory>
// #include <fftw3.h>
#include <cstring>


#include "datatypes.h"
#include "context.h"


class Reconstruction{
public:
    Reconstruction(COMPLEX* density_in, SUPPORTTYPE* support_in, std::array<std::size_t,2> dims_in): dims(dims_in), error(-1), error_real(-1){       
            
            density.resize(dims[0]*dims[1]);
            support.resize(dims[0]*dims[1]);

            memcpy(&density[0], &density_in[0], dims[0]*dims[1]*sizeof(COMPLEX));
            memcpy(&support[0], &support_in[0], dims[0]*dims[1]*sizeof(SUPPORTTYPE));


    };

    Reconstruction(std::array<std::size_t,2> dims_in): dims(dims_in), error(-1), error_real(-1){       
            
            density.resize(dims[0]*dims[1]);
            support.resize(dims[0]*dims[1]);

    };
    
    Reconstruction(const Reconstruction& rec_in): dims(rec_in.dims), error(rec_in.error), error_real(rec_in.error_real){       
            
//             printf("Copy constructor!\n");
            density.resize(dims[0]*dims[1]);
            support.resize(dims[0]*dims[1]);

            memcpy(&density[0], &(rec_in.density[0]), dims[0]*dims[1]*sizeof(COMPLEX));
            memcpy(&support[0], &(rec_in.support[0]), dims[0]*dims[1]*sizeof(SUPPORTTYPE));


    };


    
    std::vector<COMPLEX> get_density(){
        return density;
    }
    
    std::vector<SUPPORTTYPE> get_support(){
        return support;
     
    }
        
    void set_support(SUPPORTTYPE* support_in){
            memcpy(&support[0], &support_in[0], dims[0]*dims[1]*sizeof(SUPPORTTYPE));     
    }
    
    void set_density(COMPLEX* density_in){
            memcpy(&density[0], &density_in[0], dims[0]*dims[1]*sizeof(COMPLEX));
    }
    
    FLOAT get_error(){
        return error;
     
    }

    FLOAT get_fourier_error(){
        return error_fourier;
     
    }
    
        
    FLOAT get_real_error(){
        return error_real;
     
    }
    
    FLOAT get_reality(){
        return reality;
     
    }

    FLOAT get_symmetry(){
        return symmetry;
    }
    
    FLOAT get_module_sum(){
        return modulesum;
    }

    FLOAT get_complex_sum(){
        return complexsum;
    }
    
    std::vector<COMPLEX> density;
    std::vector<SUPPORTTYPE> support;
    std::array<std::size_t,2>  dims;
    FLOAT error;    
    FLOAT error_fourier;
    FLOAT error_real;
    FLOAT reality;
    FLOAT modulesum;
    FLOAT complexsum;
    FLOAT symmetry;
    
};
