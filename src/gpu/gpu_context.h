#pragma once
#include <cufft.h>
#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <memory>
#include <vector>

#include "datatypes.h"
#include "context.h"



class GPU_context: public Context{
public:
    GPU_context(int npixels_, int seed_, FLOAT* intensities,int dev_id_, int block_size_): Context(npixels_, seed_),
        dev_id(dev_id_),
        blockSize(block_size_,block_size_),
        gridSize((npixels_+block_size_-1)/block_size_, (npixels_+block_size_-1)/block_size_),
        block_size(block_size_)
        {

        cudaSetDevice(dev_id);

        cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking);
        

        if (cufftPlan2d(&plan, npixels, npixels, CUFFT_C2C) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        }
        cufftSetStream(plan, main_stream);
           
        cudaMalloc((void**)&intensities_dev, npixels * npixels *sizeof(FLOAT));
        cudaMalloc((void**)&density_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&density_ref_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&crosscorr_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&support_dev, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&buffer_dev_complex,  npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&buffer_dev_float,  npixels * npixels *sizeof(FLOAT));
        cudaMalloc((void**)&buffer_dev,  npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&error_dev, 2*sizeof(FLOAT));

        cudaMalloc((void**)&support1_dev, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&support2_dev, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&support3_dev, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&density1_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&density2_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&density3_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&densityavg_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&densitybest_dev, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMalloc((void**)&crossmap_dev, (npixels+10) * (npixels+10) *sizeof(SUPPORTTYPE));
        cudaMalloc((void**)&intensities_avg_dev, npixels * npixels *sizeof(FLOAT));

        
        cudaMallocHost((void**)&intensities_host, npixels * npixels *sizeof(FLOAT));        
        cudaMallocHost((void**)&density_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&support_host, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMallocHost((void**)&error_host, 2*sizeof(FLOAT));
        cudaMallocHost((void**)&density_ref_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&buffer_host_complex, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&buffer_host_float, npixels * npixels *sizeof(FLOAT));
        cudaMallocHost((void**)&buffer_host, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMallocHost((void**)&crosscorr_host, npixels * npixels *sizeof(CUDA_COMPLEX));

        cudaMallocHost((void**)&support1_host, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMallocHost((void**)&support2_host, npixels * npixels *sizeof(SUPPORTTYPE));
        cudaMallocHost((void**)&support3_host, npixels * npixels *sizeof(SUPPORTTYPE));        
        cudaMallocHost((void**)&density1_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&density2_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&density3_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&densityavg_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&densitybest_host, npixels * npixels *sizeof(CUDA_COMPLEX));
        cudaMallocHost((void**)&crossmap_host, (npixels+10) * (npixels+10) *sizeof(SUPPORTTYPE));
        cudaMallocHost((void**)&intensities_avg_host, npixels * npixels *sizeof(FLOAT));        

        
        
        

//         }
        
        cudaMemcpy( intensities_dev, intensities, npixels*npixels*sizeof(FLOAT), cudaMemcpyHostToDevice);
    }
    
    
    ~GPU_context(){

        cufftDestroy(plan);

        cudaStreamDestroy(main_stream);
        

        cudaFree(intensities_dev);
        cudaFree(density_dev);
        cudaFree(density_ref_dev);
        cudaFree(crosscorr_dev);
        cudaFree(support_dev);
        cudaFree(buffer_dev_complex);
        cudaFree(buffer_dev_float);
        cudaFree(buffer_dev);
        cudaFree(error_dev);
        cudaFree(density1_dev);
        cudaFree(density2_dev);
        cudaFree(density3_dev);
        cudaFree(densityavg_dev);
        cudaFree(densitybest_dev);
        cudaFree(support1_dev);
        cudaFree(support2_dev);
        cudaFree(support3_dev);
        cudaFree(crossmap_dev);
        cudaFree(intensities_avg_dev);



        cudaFreeHost(intensities_host);
        cudaFreeHost(density_host);
        cudaFreeHost(density_ref_host);
        cudaFreeHost(buffer_host_complex);
        cudaFreeHost(buffer_host_float);
        cudaFreeHost(buffer_host);
        cudaFreeHost(crosscorr_host);
        cudaFreeHost(support_host);
        cudaFreeHost(error_host);
        cudaFreeHost(density1_host);
        cudaFreeHost(density2_host);
        cudaFreeHost(density3_host);
        cudaFreeHost(densityavg_host);
        cudaFreeHost(densitybest_host);
        cudaFreeHost(support1_host);
        cudaFreeHost(support2_host);
        cudaFreeHost(support3_host);
        cudaFreeHost(crossmap_host);
        cudaFreeHost(intensities_avg_host);
 
        
    }
    
    void get_ft();
    void get_ift();
    void get_ift(CUDA_COMPLEX*);
    
    void set_support(SUPPORTTYPE* support);
    void set_density(COMPLEX* density);
    
    void set_buffer(COMPLEX* support);
    void set_buffer(FLOAT* data);
    void set_buffer(SUPPORTTYPE* support);
    
    void get_buffer(COMPLEX* support);
    void get_buffer(FLOAT* data);
    void get_buffer(SUPPORTTYPE* support);
    
    void add_to_buffer(SUPPORTTYPE* support);
    void add_to_buffer(COMPLEX* density);
    void add_module_to_buffer(COMPLEX* density, bool ft);
    
    
    
    void set_ref_density(COMPLEX* density);
    void get_density(COMPLEX* density);
    void get_support(SUPPORTTYPE* density);   
    void get_pattern(FLOAT*);   
    void set_pattern(FLOAT*);   
    
    void get_shift(int& shiftx, int& shifty, FLOAT& shiftphase, bool& flip, FLOAT minr, FLOAT maxr);
    void reshift(int shiftx, int shifty, FLOAT shiftphase, bool flip);
    
    void HIO(int niterations, FLOAT beta, FLOAT real);
    void RAAR(int niterations, FLOAT beta, FLOAT real);

    void ER(int niterations, FLOAT real);

    void SW(FLOAT sigma, FLOAT threshold, FLOAT real);
    void SWA(FLOAT sigma, int spixels, FLOAT real);

    void get_error( FLOAT real, FLOAT &error_fourier, FLOAT &error_real, FLOAT &complexsum, FLOAT &modulesum);
    
// // // // //     
    
    void set_parents_support(SUPPORTTYPE* support1, SUPPORTTYPE* support2, SUPPORTTYPE* support3);
    void set_parents_density(COMPLEX* density1, COMPLEX* density2, COMPLEX* density3, COMPLEX* densityavg, COMPLEX* densitybest);
    void get_parents_ft();
    void prepare_parents(FLOAT minr, FLOAT maxr);
    void reshift_parent(CUDA_COMPLEX* densitypar_host, CUDA_COMPLEX* densitypar_dev, SUPPORTTYPE* supportpar_dev,FLOAT minr, FLOAT maxr);
    void set_avg_intensities(FLOAT* intensities_avg);
    void get_new_individual(SUPPORTTYPE* crossmap, int crosssizex, int crosssizey,int squaresizex, int squaresizey,  int offsetx, int offsety, FLOAT rotangle, FLOAT crossdiff, FLOAT use_average, bool crossft, bool crosssym, FLOAT real);
    
    void prepare_average();
    void normalize_parents(FLOAT crossdiff);
    
    void get_symmetry(FLOAT& symmetry, FLOAT symexp,FLOAT symmin, FLOAT symmax);
    
    
// // // // // // // // // // // // // // // // // //     
    
    
    
    FLOAT* error_dev;
    FLOAT* error_host;
    
    
    
    int block_size;
    cudaStream_t main_stream;
    cufftHandle plan;


    FLOAT* intensities_dev;
    FLOAT* intensities_host;

    FLOAT* intensities_avg_dev;
    FLOAT* intensities_avg_host;
    
    CUDA_COMPLEX* density_dev;
    CUDA_COMPLEX* density_ref_dev;
    CUDA_COMPLEX* crosscorr_dev;
    SUPPORTTYPE* support_dev;
    
    CUDA_COMPLEX* buffer_dev_complex;
    CUDA_COMPLEX* buffer_host_complex;
    FLOAT* buffer_dev_float;
    FLOAT* buffer_host_float;
    SUPPORTTYPE* buffer_dev;
    SUPPORTTYPE* buffer_host;
    
    CUDA_COMPLEX* density_host;
    CUDA_COMPLEX* density_ref_host;
    CUDA_COMPLEX* crosscorr_host;

    SUPPORTTYPE* support_host;

    SUPPORTTYPE* support1_host;
    SUPPORTTYPE* support2_host;
    SUPPORTTYPE* support3_host;
    CUDA_COMPLEX* density1_host;
    CUDA_COMPLEX* density2_host;
    CUDA_COMPLEX* density3_host;
    CUDA_COMPLEX* densityavg_host;
    CUDA_COMPLEX* densitybest_host;

    
    SUPPORTTYPE* support1_dev;
    SUPPORTTYPE* support2_dev;
    SUPPORTTYPE* support3_dev;
    CUDA_COMPLEX* density1_dev;
    CUDA_COMPLEX* density2_dev;
    CUDA_COMPLEX* density3_dev;

    CUDA_COMPLEX* densityavg_dev;
    CUDA_COMPLEX* densitybest_dev;

    
    SUPPORTTYPE* crossmap_host;
    SUPPORTTYPE* crossmap_dev;

    

    CUDA_COMPLEX density_ref_sum;

    
    dim3 blockSize;
    dim3 gridSize;
    
    int dev_id;


    
};



