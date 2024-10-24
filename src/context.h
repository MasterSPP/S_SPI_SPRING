#pragma once


#include "datatypes.h"
#include <random>





class Context{
public:
    Context(int npixels_, int seed): npixels(npixels_), generator(seed), distribution(0,1){}    
    
    ~Context(){}

    FLOAT rand(FLOAT min, FLOAT max){return distribution(generator)*(max-min)+min;};
    FLOAT rand(){return distribution(generator);};

    
    virtual void get_ft(){};
    virtual void get_ift(){};
    virtual void set_support(SUPPORTTYPE* support){};
    virtual void set_density(COMPLEX* density){};
    virtual void get_density(COMPLEX* density){};   
    virtual void get_support(SUPPORTTYPE* density){};  
    
    virtual void set_buffer(COMPLEX* support){};
    virtual void set_buffer(FLOAT* data){};
    virtual void set_buffer(SUPPORTTYPE* support){};
    
    virtual void get_buffer(COMPLEX* support){};
    virtual void get_buffer(FLOAT* data){};
    virtual void get_buffer(SUPPORTTYPE* support){};
    
    virtual void add_to_buffer(COMPLEX* density){};
    virtual void add_to_buffer(SUPPORTTYPE* density){};
    virtual void add_module_to_buffer(COMPLEX* density, bool ft){};
    
    
    virtual void get_pattern(FLOAT* intensity){};   
    virtual void set_pattern(FLOAT* intensity){};   
    virtual void set_ref_density(COMPLEX* refdensity){};
    virtual void get_shift(int& shiftx, int& shifty, FLOAT& shiftphase, bool& flip, FLOAT minr, FLOAT maxr){};
    virtual void reshift(int shiftx, int shifty, FLOAT shiftphase, bool flip){} ;
    
    virtual void HIO(int niterations, FLOAT beta, FLOAT real){};
    virtual void RAAR(int niterations, FLOAT beta, FLOAT real){};

    virtual void ER(int niterations,  FLOAT real){};

    virtual void SW(FLOAT sigma, FLOAT threshold, FLOAT real){};
    virtual void SWA(FLOAT sigma, int spixels, FLOAT real){};

// // // // // //     
    virtual void set_parents_support(SUPPORTTYPE* support1, SUPPORTTYPE* support2, SUPPORTTYPE* support3){};
    virtual void set_parents_density(COMPLEX* density1, COMPLEX* density2, COMPLEX* density3, COMPLEX* densityavg, COMPLEX* densitybest){};
    virtual void get_parents_ft(){};
    virtual void prepare_parents(FLOAT minr, FLOAT maxr){};
    virtual void set_avg_intensities(FLOAT* intensities_avg){};
    virtual void get_new_individual(SUPPORTTYPE* crossmap, int crosssizex, int crosssizey, int squaresizex, int squaresizey, int offsetx, int offsety, FLOAT rotangle, FLOAT crossdiff, FLOAT use_average, bool crossft, bool crosssym, FLOAT real){};
    
    
    virtual void get_error( FLOAT real, FLOAT &error_fourier, FLOAT &error_real, FLOAT &complexsum, FLOAT &sum){};
    virtual void get_symmetry(FLOAT& symmetry, FLOAT symexp,FLOAT symmin, FLOAT symmax){};

    int npixels;
    std::mt19937 generator;
    std::uniform_real_distribution<FLOAT> distribution;
};
