#pragma once
#include <cstdint>
#include <complex>

#define FLOAT            float
#define FFTW_COMPLEX     fftwf_complex
#define CUDA_COMPLEX     cufftComplex
#define COMPLEX          std::complex<FLOAT>


#define SUPPORTTYPE      int
#define fftw_malloc         fftwf_malloc
#define fftw_plan           fftwf_plan
#define fftw_plan_dft_2d    fftwf_plan_dft_2d
#define fftw_execute        fftwf_execute
#define fftw_free           fftwf_free
#define fftw_destroy_plan   fftwf_destroy_plan
#define fftw_execute_dft    fftwf_execute_dft
typedef long long int llint;
