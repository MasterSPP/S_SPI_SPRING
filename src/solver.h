#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <memory>
// #include <fftw3.h>
#include <functional>
#include <iostream>
#include <algorithm>
#include <tuple>
#include "datatypes.h"
#include "context.h"

#include "reconstruction.h"

class Solver{
public:
    Solver(FLOAT* pattern_in, std::array<std::size_t,2> dims_in, std::array<int,2> center_in, FLOAT reality_, FLOAT realpen_, int nthreads_, bool use_bounds_, std::vector<int> GPU_ids, int seed);

    ~Solver();
    
    void prepare(bool);
    
    void set_threads(int nthreads_=0);
    
    std::vector<std::shared_ptr<Reconstruction>>  HIO(std::vector<std::shared_ptr<Reconstruction>>  rec, int iterations, FLOAT beta);
    std::vector<std::shared_ptr<Reconstruction>>  RAAR(std::vector<std::shared_ptr<Reconstruction>>  rec, int iterations, FLOAT beta);

    std::vector<std::shared_ptr<Reconstruction>>   ER(std::vector<std::shared_ptr<Reconstruction>>  rec, int iterations);
    std::vector<std::shared_ptr<Reconstruction>> SW( std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT sigma, FLOAT threshold);
    std::vector<std::shared_ptr<Reconstruction>> SWA( std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT sigma, int pixels);
    
    std::vector<FLOAT>  get_profile(std::shared_ptr<Reconstruction> rec);
     
    void Evaluate(std::vector<std::shared_ptr<Reconstruction>> rec);
    std::vector<std::shared_ptr<Reconstruction>>  Sequence(std::vector<std::shared_ptr<Reconstruction>> rec, int nrep);
    std::vector<std::shared_ptr<Reconstruction>>  Reshift(std::vector<std::shared_ptr<Reconstruction>> rec_array, std::shared_ptr<Reconstruction> ref);
    
    std::shared_ptr<Reconstruction>  Reshift_rec(std::shared_ptr<Reconstruction> rec, std::shared_ptr<Reconstruction> ref, int icontext);
    std::vector<std::shared_ptr<Reconstruction>>  Crossover(std::vector<std::shared_ptr<Reconstruction>> rec_array, std::shared_ptr<Reconstruction> best_rec,std::shared_ptr<Reconstruction> midrec, FLOAT crossbalance, FLOAT crossdiff, FLOAT bestfraction, FLOAT crossft, bool crosssym, FLOAT use_best, bool update_support);
    
    std::vector<FLOAT>  get_pattern();
    void set_sequence(std::vector<std::tuple<std::string, std::vector<FLOAT>>> sequence);
    
    std::function<void(Context*)> get_algorithm(std::tuple<std::string, std::vector<FLOAT>> alg);
    
    std::shared_ptr<Reconstruction> GetAverage(std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT fraction);
    std::vector<FLOAT> GetAverageModule(std::vector<std::shared_ptr<Reconstruction>> rec_array, bool ft, FLOAT fraction);
        
    void set_reality(FLOAT reality_){ reality=reality_; }

    void set_shift_param(FLOAT minr_, FLOAT maxr_);

    void set_pattern(FLOAT* pattern_in);

    void set_boundaries();

    // // // // // // // // // // // // // // // 
    
    std::shared_ptr<Reconstruction>  Initialize_rec(std::shared_ptr<Reconstruction> rec, int support_size,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems_in,  std::array<FLOAT,2> exponent_in,  std::array<FLOAT,2> scaling_in, FLOAT phase_in,  FLOAT suppcut, int icontext);
    std::shared_ptr<Reconstruction>  Initialize_rec(std::shared_ptr<Reconstruction> rec, std::shared_ptr<Reconstruction> refrec,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems_in,  std::array<FLOAT,2> exponent_in,  std::array<FLOAT,2> scaling_in, FLOAT phase_in,  FLOAT suppcut, int icontext);


    std::shared_ptr<Reconstruction>  Initialize_rec(std::shared_ptr<Reconstruction> rec, std::vector<SUPPORTTYPE> refsupport,  FLOAT initfrac, std::array<FLOAT,2> scaling_in, FLOAT phase_in, int icontext);
    
    std::vector<std::shared_ptr<Reconstruction>>  Initialize_pop(int nrec, int support_size, std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling_in, FLOAT phase_in,  FLOAT suppcut);
    std::vector<std::shared_ptr<Reconstruction>>  Initialize_pop(int nrec, std::shared_ptr<Reconstruction> refrec,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling_in, FLOAT phase_in,  FLOAT suppcut);

    
    std::vector<std::shared_ptr<Reconstruction>>  Initialize_pop(int nrec, std::vector<SUPPORTTYPE> refsupport,FLOAT initfrac, std::array<FLOAT,2> scaling_in, FLOAT phase_in);
    
// // // // //     
    
    std::vector<Context*> contexts;
    
    std::vector<FLOAT> pattern;

    std::array<std::size_t,2> dims;
    std::array<int,2> center;

    std::size_t size;
    
    std::vector<std::function<void(Context*)>> algorithms_seq;

    FLOAT reality;
    FLOAT shiftmin, shiftmax;
    FLOAT realpen;
    int nthreads;

};
