#include "gpu/gpu_context.h"
#include <algorithm>



__global__ void get_module(CUDA_COMPLEX* density, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        FLOAT mod = std::sqrt(density[i].x*density[i].x + density[i].y*density[i].y);
        density[i].x=mod;
	density[i].y=0;
    }
    
}



__global__ void normalize_shift(CUDA_COMPLEX* density, FLOAT normexp, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        FLOAT mod = std::sqrt(density[i].x*density[i].x + density[i].y*density[i].y);
        FLOAT newmod = std::pow(mod, normexp);
        density[i].x=newmod;
        density[i].y=0;
//         
    }
    
}


__global__ void normalize_ft_shift(CUDA_COMPLEX* density, FLOAT normexp, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        FLOAT mod = std::sqrt(density[i].x*density[i].x + density[i].y*density[i].y);
        FLOAT newmod = std::pow(mod, normexp);
        density[i].x*=newmod/mod;
        density[i].y*=newmod/mod;
        
    }
    
}

__device__ __forceinline__ FLOAT atomicMax(FLOAT *address, FLOAT val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}


__global__ void normalize(CUDA_COMPLEX* data, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            data[index].x/=FLOAT(npixels*npixels);
            data[index].y/=FLOAT(npixels*npixels);

        
    }
    
}

__global__ void sub_intensities(CUDA_COMPLEX* data, FLOAT* intensities, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            
            if(intensities[index]>=0.f){    //se il pixel del pattern e' noto
                if(m>0.f){
                    data[index].x = intensities[index]*data[index].x/(m);        // write on the input the experimental module times the real ...
                    data[index].y = intensities[index]*data[index].y/(m);        // and the imaginary part of exp(i x phase)
                }
                else{
                    FLOAT phase=std::atan2(data[index].y,data[index].x);
                    data[index].x=intensities[index]*cos(phase);
                    data[index].y=intensities[index]*sin(phase);
                }
            }
            
            else if(intensities[index]<-1.5f and m>-intensities[index]-2.f){
                    FLOAT newmod = -intensities[index]-2.f;
                    data[index].x = newmod*data[index].x/(m);        // write on the input the experimental module times the real ...
                    data[index].y = newmod*data[index].y/(m); 
                    
                }
                
            
        
    }
    
}


__global__ void add_module(CUDA_COMPLEX* data, FLOAT* datasum, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            datasum[index]+=std::sqrt(data[index].x*data[index].x + data[index].y*data[index].y);
    }
    
}

__global__ void add(CUDA_COMPLEX* data, CUDA_COMPLEX* datasum, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            datasum[index].x+=data[index].x;
            datasum[index].y+=data[index].y;        
    }
    
}


__global__ void normalize(CUDA_COMPLEX* data, FLOAT normfactor, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            data[index].x*=normfactor;
            data[index].y*=normfactor;

        
    }
    
}


__global__ void normalize(FLOAT* data, FLOAT normfactor, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            data[index]*=normfactor;        
    }
    
}


__global__ void sub_module(CUDA_COMPLEX* data, FLOAT* newmod, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            
            if(m>0){            
                data[index].x = newmod[index]*data[index].x/(m);        // write on the input the experimental module times the real ...
                data[index].y = newmod[index]*data[index].y/(m);        // and the imaginary part of exp(i x phase)
            }
        
    }
    
}




__global__ void add(SUPPORTTYPE* data, SUPPORTTYPE* datasum, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            datasum[index]+=data[index];
    }
    
}



__global__ void smooth_intensities(CUDA_COMPLEX* data, FLOAT sigma, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            
            int xx = ix;
            int yy = iy;
            if(xx>=npixels/2) xx-=npixels;
            if(yy>=npixels/2) yy-=npixels;
            
            FLOAT square_radius = xx*xx+yy*yy;
            
            FLOAT coeff = exp( -square_radius/(sigma*sigma));
            data[index].x*=coeff;
            data[index].y*=coeff;        
    }
    
}
__global__ void get_module_sum(CUDA_COMPLEX* data, FLOAT* sum, FLOAT exp,int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
                            
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            if(m>0){
                atomicAdd(&local_sum,std::pow(m, exp));   
            }
            
    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&sum[0],local_sum);
    }
    
}


__global__ void get_module_sum(CUDA_COMPLEX* data, FLOAT* intensities, FLOAT* sum, FLOAT exp,int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum;
    __shared__ FLOAT local_sum_int;

    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum=0;
        local_sum_int=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
                            
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            FLOAT mint = intensities[index];
            if(intensities[index]>=0){
                atomicAdd(&local_sum,std::pow(m, exp));   
                atomicAdd(&local_sum_int,std::pow(mint, exp));                   
            }
            
    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&sum[0],local_sum);
        atomicAdd(&sum[1],local_sum_int);

        
    }
    
}



__global__ void get_real_sum(FLOAT* data, FLOAT* sum, FLOAT exp,int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
                            
            FLOAT m=data[index];
            if(m>0){
                atomicAdd(&local_sum,std::pow(m, exp));   
            }
            
    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&sum[0],local_sum);
    }
    
}


__global__ void get_complex_sum(CUDA_COMPLEX* data, FLOAT* sum, FLOAT exp, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum_r, local_sum_i;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum_r=0;
        local_sum_i=0;
        
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
                            
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            if(m>0){
                FLOAT newm = std::pow(m, exp);      
                atomicAdd(&local_sum_r,data[index].x/m*newm);   
                atomicAdd(&local_sum_i,data[index].y/m*newm);   
            }
    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&sum[0],local_sum_r);
        atomicAdd(&sum[1],local_sum_i);        
    }
    
}


__global__ void get_error_gpu(CUDA_COMPLEX* data, FLOAT* intensities, int npixels, FLOAT* error){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum, local_error;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum=0;
        local_error=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            
            if(intensities[index]>=0){
                
                atomicAdd(&local_sum,intensities[index]*intensities[index]);   
                            
                FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
                atomicAdd(&local_error,(intensities[index]-m)*(intensities[index]-m));   
 
            }

    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&error[0],local_error);
        atomicAdd(&error[1],local_sum);
    }
    
}
__global__ void get_real_error_gpu(CUDA_COMPLEX* data, SUPPORTTYPE* support, FLOAT real_flag,  int npixels, FLOAT* error){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_sum, local_error;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_sum=0;
        local_error=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            
             CUDA_COMPLEX datasupp;
             datasupp.x = data[index].x;
             datasupp.y = data[index].y;
                         
             FLOAT tempsupp = support[index];
             FLOAT phase = atan2(datasupp.y, datasupp.x);
             if(phase>(1.f-real_flag)*M_PI or phase<-(1.f-real_flag)*M_PI) tempsupp=0.f;
//             
             datasupp.x=datasupp.x*tempsupp;
             datasupp.y=datasupp.y*tempsupp;
            
            
             
             FLOAT normval = data[index].y*data[index].y+data[index].x*data[index].x;
             atomicAdd(&local_sum,normval);   
             
             FLOAT diff = (data[index].y-datasupp.y)*(data[index].y-datasupp.y)
                         +(data[index].x*datasupp.x)*(data[index].x*datasupp.x);
             
             
                if(support[index]==0){
                    atomicAdd(&local_error,diff);   
                }

    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicAdd(&error[0],local_error);
        atomicAdd(&error[1],local_sum);
    }
    
}
__global__ void get_max_module(CUDA_COMPLEX* data, FLOAT* maxbuffer, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ FLOAT local_max;
    if(threadIdx.x ==0 && threadIdx.y==0){
        local_max=0;
    }
    
    __syncthreads();
    
    if(ix<npixels && iy<npixels){
            const int index = ix+iy*npixels;
            FLOAT m=std::sqrt(data[index].y*data[index].y+data[index].x*data[index].x);
            
            if(m>=local_max)
                atomicMax(&local_max, m);   

    }
    __syncthreads();

    
    if(threadIdx.x ==0 && threadIdx.y==0){
        atomicMax(&maxbuffer[0],local_max);
    }
    
}
    
    


__global__ void cut_support(CUDA_COMPLEX* data, SUPPORTTYPE* support, FLOAT threshold, FLOAT* maxbuffer, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;
            FLOAT m=std::sqrt(data[i].y*data[i].y+data[i].x*data[i].x);
            
            if(m>=maxbuffer[0]*threshold)
                support[i]=1;
            else
                support[i]=0;
        
    }
    
}


__global__ void cut_support(CUDA_COMPLEX* data, SUPPORTTYPE* support, FLOAT threshold,int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;
            FLOAT m=std::sqrt(data[i].y*data[i].y+data[i].x*data[i].x);
            
            if(m>=threshold)
                support[i]=1;
            else
                support[i]=0;
        
    }
    
}


__global__ void cut_density(CUDA_COMPLEX* data,FLOAT threshold,int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;
            FLOAT m=std::sqrt(data[i].y*data[i].y+data[i].x*data[i].x)-threshold;
            if(m<0) m=0;
            
            data[i].x=m;
            data[i].y=0;
        
    }
    
}


__global__ void apply_support_er(CUDA_COMPLEX* data, SUPPORTTYPE* support,int npixels, FLOAT real_flag){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;

            FLOAT tempsupp = support[i];
             
             FLOAT phase = atan2(data[i].y, data[i].x);

             if(phase>(1.f-real_flag)*M_PI or phase<-(1.f-real_flag)*M_PI) tempsupp=0.f;

             data[i].x=data[i].x*tempsupp;
             data[i].y=data[i].y*tempsupp;
        
    }
    
}


__global__ void apply_support_hio(CUDA_COMPLEX* data, SUPPORTTYPE* support,CUDA_COMPLEX* buffer_data,  FLOAT beta, int npixels,  FLOAT real_flag){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    
    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;


            FLOAT tempsupp = support[i];
            
            FLOAT phase = atan2(data[i].y, data[i].x);
            
            if(phase>(1.f-real_flag)*M_PI or phase<-(1.f-real_flag)*M_PI) tempsupp=0.f;
            
            data[i].x=(buffer_data[i].x-beta*data[i].x)*(1.f-tempsupp) + data[i].x*tempsupp;
            data[i].y=(buffer_data[i].y-beta*data[i].y)*(1.f-tempsupp) + data[i].y*tempsupp;
            
            
            buffer_data[i].x=data[i].x;
            buffer_data[i].y=data[i].y;
          
        
    }
    
}


__global__ void apply_support_raar(CUDA_COMPLEX* data, SUPPORTTYPE* support,CUDA_COMPLEX* buffer_data,  FLOAT beta, int npixels,  FLOAT real_flag){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    
    if(ix<npixels && iy<npixels){
            const int i = ix+iy*npixels;
        
            CUDA_COMPLEX reflval;
            reflval.x = 2.f*data[i].x-buffer_data[i].x;
            reflval.y = 2.f*data[i].y-buffer_data[i].y;

            FLOAT phase = atan2(reflval.y, reflval.x);
            
            if(phase>(1.-real_flag)*M_PI or phase<-(1.f-real_flag)*M_PI or support[i]==0){
                data[i].x=(beta*buffer_data[i].x+(1.f-2.f*beta)*data[i].x);
                data[i].y=(beta*buffer_data[i].y+(1.f-2.f*beta)*data[i].y);              
                
            }

            buffer_data[i].x=data[i].x;
            buffer_data[i].y=data[i].y;
               
    }
    
}


    
    
    void GPU_context::set_support(SUPPORTTYPE* support){
        cudaSetDevice(dev_id);
        memcpy(support_host, support, npixels*npixels*sizeof(SUPPORTTYPE));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( support_dev, support_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);
        cudaStreamSynchronize(main_stream);

        
    }
    
    void GPU_context::set_pattern(FLOAT* intensities_){
        cudaSetDevice(dev_id);
        memcpy(intensities_host, intensities_, npixels*npixels*sizeof(FLOAT));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( intensities_dev, intensities_host, npixels*npixels*sizeof(FLOAT), 
                        cudaMemcpyHostToDevice, main_stream);
        cudaStreamSynchronize(main_stream);

        
    }
    void GPU_context::set_density(COMPLEX* density){
        cudaSetDevice(dev_id);
        memcpy(density_host, density, npixels*npixels*sizeof(CUDA_COMPLEX));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( density_dev, density_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);     
        cudaStreamSynchronize(main_stream);

    }
    
    void GPU_context::set_ref_density(COMPLEX* density){
        cudaSetDevice(dev_id);
        memcpy(density_ref_host, density, npixels*npixels*sizeof(CUDA_COMPLEX));
        
        density_ref_sum.x=0;
        density_ref_sum.y=0;
        for(int i=0; i<npixels*npixels; i++){
            density_ref_sum.x+=density_ref_host[i].x;
            density_ref_sum.y+=density_ref_host[i].y;
        }
        
        
        cudaStreamSynchronize(main_stream);

        

        cudaMemcpyAsync( density_ref_dev, density_ref_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);    
        
        
    
        normalize_shift<<<gridSize, blockSize,0,main_stream>>>(density_ref_dev,1., npixels);

              
        
// // // // // // // 
        
        if (cufftExecC2C(plan, density_ref_dev, density_ref_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }

        normalize_ft_shift<<<gridSize, blockSize,0,main_stream>>>(density_ref_dev, 1, npixels);

        cudaStreamSynchronize(main_stream);

    }
        
    void GPU_context::get_pattern(FLOAT* intensities){
        cudaSetDevice(dev_id);
        cudaMemcpy( intensities, intensities_dev, npixels*npixels*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost);        
        
        }
    
    void GPU_context::get_density(COMPLEX* density){
        cudaSetDevice(dev_id);
        cudaMemcpyAsync( density_host, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyDeviceToHost, main_stream);        
        cudaStreamSynchronize(main_stream);

        memcpy(density, density_host, npixels*npixels*sizeof(CUDA_COMPLEX));
    }
    
    void GPU_context::get_support(SUPPORTTYPE* support){
        cudaSetDevice(dev_id);
        cudaMemcpyAsync( support_host, support_dev, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyDeviceToHost, main_stream);        
        cudaStreamSynchronize(main_stream);

        memcpy(support, support_host, npixels*npixels*sizeof(SUPPORTTYPE));
    }
    
    
    void GPU_context::set_buffer(COMPLEX* data){
        cudaSetDevice(dev_id);
        memcpy(buffer_host_complex, data, npixels*npixels*sizeof(CUDA_COMPLEX));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( buffer_dev_complex, buffer_host_complex, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);     
        cudaStreamSynchronize(main_stream);
    }
    void GPU_context::set_buffer(FLOAT* data){
        cudaSetDevice(dev_id);
        memcpy(buffer_host_float, data, npixels*npixels*sizeof(FLOAT));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( buffer_dev_float, buffer_host_float, npixels*npixels*sizeof(FLOAT), 
                        cudaMemcpyHostToDevice, main_stream);     
        cudaStreamSynchronize(main_stream);
    }
    void GPU_context::set_buffer(SUPPORTTYPE* data){
        cudaSetDevice(dev_id);
        memcpy(buffer_host, data, npixels*npixels*sizeof(SUPPORTTYPE));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( buffer_dev, buffer_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);     
        cudaStreamSynchronize(main_stream);
    }

    
    
    void GPU_context::get_buffer(COMPLEX* data){
        cudaSetDevice(dev_id);
        cudaMemcpyAsync( buffer_host_complex, buffer_dev_complex, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyDeviceToHost, main_stream);        
        cudaStreamSynchronize(main_stream);
        memcpy(data, buffer_host_complex, npixels*npixels*sizeof(CUDA_COMPLEX));
    }    
    void GPU_context::get_buffer(FLOAT* data){
        cudaSetDevice(dev_id);
        cudaMemcpyAsync( buffer_host_float, buffer_dev_float, npixels*npixels*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);        
        cudaStreamSynchronize(main_stream);
        memcpy(data, buffer_host_float, npixels*npixels*sizeof(FLOAT));
    }
    void GPU_context::get_buffer(SUPPORTTYPE* data){
        cudaSetDevice(dev_id);
        cudaMemcpyAsync( buffer_host, buffer_dev, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyDeviceToHost, main_stream);        
        cudaStreamSynchronize(main_stream);
        memcpy(data, buffer_host, npixels*npixels*sizeof(SUPPORTTYPE));
    }
    
    
    void GPU_context::get_ft(){
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            } 
                
    }
    
    
    void GPU_context::get_ift(){
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
                
    }
    
        
    void GPU_context::get_ift(CUDA_COMPLEX* density){
            
            if (cufftExecC2C(plan, density, density, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density, npixels);
                
    }
  
    
    
     void GPU_context::HIO(int niterations, FLOAT beta,   FLOAT real){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream);  
        
        
        for(int i_iteration=0; i_iteration<niterations; i_iteration++){
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            
            sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
            
            apply_support_hio<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, buffer_dev_complex, beta, npixels,  real);
            
        }
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }

        sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 

        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
        
        
        cudaStreamSynchronize(main_stream);  
         
    }
  
  
     void GPU_context::RAAR(int niterations, FLOAT beta,   FLOAT real){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream);  
        
        
        for(int i_iteration=0; i_iteration<niterations; i_iteration++){
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            
            sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
            
            apply_support_raar<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, buffer_dev_complex, beta, npixels, real);
            
        }
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }

        sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 

        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
        
        
        cudaStreamSynchronize(main_stream);  
         
    }

    
    
    void GPU_context::ER(int niterations, FLOAT real){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);

        
        for(int i_iteration=0; i_iteration<niterations; i_iteration++){
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            
            sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
            
            apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, npixels, real);
            
        }
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }

        sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 

        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
        
        
        
        
        
        cudaStreamSynchronize(main_stream);

    }
    
    
    
    void GPU_context::SW(FLOAT sigma, FLOAT threshold, FLOAT real){
        cudaStreamSynchronize(main_stream);

        cudaSetDevice(dev_id);
        
        cudaMemcpyAsync( buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
///////         

	if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            
            sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
            

//////
         get_module<<<gridSize, blockSize,0,main_stream>>>(density_dev,  npixels); 
   
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        
        
        FLOAT sigmaft = FLOAT(npixels)/sigma/M_PI;
        smooth_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, sigmaft,  npixels); 
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
            
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    

        get_max_module<<<gridSize, blockSize,0,main_stream>>>(density_dev, error_dev, npixels); 

        cut_support<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, threshold, error_dev, npixels);
            
        cudaMemcpyAsync( density_dev, buffer_dev_complex, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
        cudaStreamSynchronize(main_stream);

    }          
    
    
    
    void GPU_context::SWA(FLOAT sigma, int spixels,   FLOAT real){
        cudaStreamSynchronize(main_stream);

        cudaSetDevice(dev_id);
        
        cudaMemcpyAsync( buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 

///////         

	if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            
            sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
            
            if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            }
            normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
            

//////
 



         get_module<<<gridSize, blockSize,0,main_stream>>>(density_dev,  npixels); 
   

        
		
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        
        
        FLOAT sigmaft = FLOAT(npixels)/sigma/M_PI;
        smooth_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, sigmaft,  npixels); 
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        
        cudaMemcpyAsync( buffer_host_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToHost, main_stream); 
        
        std::vector<FLOAT> temp(npixels*npixels);
        
        cudaStreamSynchronize(main_stream);
        
        FLOAT maxval=0;
        for(int index=0; index<npixels*npixels; index++){
            temp[index]=std::sqrt(buffer_host_complex[index].y*buffer_host_complex[index].y+buffer_host_complex[index].x*buffer_host_complex[index].x);
            if(temp[index]>maxval) maxval=temp[index];
        }
        std::sort(temp.begin(), temp.end());
        if(spixels>npixels*npixels) spixels=npixels*npixels;
        FLOAT threshold = temp[npixels*npixels-spixels];        
//         printf("SWA\n %.5f %d", threshold, spixels);
        cudaStreamSynchronize(main_stream);


        cut_support<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, threshold, npixels);
            
        cudaMemcpyAsync( density_dev, buffer_dev_complex, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
        cudaStreamSynchronize(main_stream);

    }          
    
   
    void GPU_context::get_error( FLOAT real, FLOAT &error_fourier_val, FLOAT &error_real_val, FLOAT &complexsum, FLOAT &modulesum){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);

        FLOAT error_fourier, error_fourier_norm, fourier_max, module_sum, error_real, error_real_norm;
        
        cudaMemcpyAsync( buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream);  
        
        
        apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, npixels, real);

        
        cudaMemcpyAsync( crosscorr_dev, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream);  

        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    

        
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        
        get_error_gpu<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels, error_dev); 
        
        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        cudaStreamSynchronize(main_stream);



        error_fourier = error_host[0];
        error_fourier_norm = error_host[1];
        

        
// // // // // // // // // // // // // // // // // // // // // // // // // //         
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_max_module<<<gridSize, blockSize,0,main_stream>>>(density_dev, error_dev, npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        fourier_max = error_host[0];
        
        
// // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, error_dev, 1. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        module_sum = error_host[0];

        
        
// // // // // // // // // // // // // // // // // // // //         d
        cudaStreamSynchronize(main_stream);

        sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }
        normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev, npixels);
        
        
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_real_error_gpu<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, real, npixels, error_dev); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        cudaStreamSynchronize(main_stream);

        error_real = error_host[0];
        error_real_norm = error_host[1];
        
// // // // // // // // // // // // // // // // // // // // 

        
        cudaMemcpyAsync( density_dev, buffer_dev_complex, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
        
        cudaStreamSynchronize(main_stream);

        
        error_fourier_val = std::pow(error_fourier/error_fourier_norm,0.5);

        error_real_val = std::pow(error_real/error_real_norm,0.5);
        
        complexsum = fourier_max;
        modulesum = module_sum;
                
        
    }
    
    



    
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // 


__global__ void multiply(CUDA_COMPLEX* crosscorr, CUDA_COMPLEX* density,CUDA_COMPLEX* refdensity, FLOAT sign, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        crosscorr[i].x = density[i].x*refdensity[i].x - (sign*density[i].y)*refdensity[i].y;
        crosscorr[i].y = density[i].x*refdensity[i].y + (sign*density[i].y)*refdensity[i].x;
    
    }
    
}
            
__global__ void multiply(CUDA_COMPLEX* crosscorr, CUDA_COMPLEX* density,CUDA_COMPLEX* refdensity, FLOAT* intensities, FLOAT minr, FLOAT maxr, FLOAT sign, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        FLOAT x = ix;
        FLOAT y = iy;
        if(x>=npixels/2) x-=npixels;
        if(y>=npixels/2) y-=npixels;
        crosscorr[i].x = density[i].x*refdensity[i].x - (sign*density[i].y)*refdensity[i].y;
        crosscorr[i].y = density[i].x*refdensity[i].y + (sign*density[i].y)*refdensity[i].x;
        
        FLOAT radius = std::sqrt(x*x+y*y); 
        FLOAT sigmatot =  FLOAT(npixels)*minr/2.+1e-12;
        FLOAT sigma = sigmatot/10.;
        FLOAT effradius = radius-(sigmatot-sigma);
              
        
        FLOAT coeff = effradius>0 ? 1.-exp( -effradius*effradius/(sigma*sigma)) : 0.;
        
        
        FLOAT sigmatotmax =  FLOAT(npixels)*maxr/2.+1e-12;
        FLOAT sigmamax = sigmatotmax/10.;
        FLOAT effradiusmax = radius-(sigmatotmax-sigmamax);
        
        
        FLOAT coeffmax = effradiusmax>0 ? exp( -effradiusmax*effradiusmax/(sigmamax*sigmamax)) : 1.;
        
        coeff*=coeffmax;

        crosscorr[i].x *= coeff*coeff;
        crosscorr[i].y *= coeff*coeff;
        
        if(intensities[i]<0){
            crosscorr[i].x = 0; 
            crosscorr[i].y = 0;
        }
        
        

        }
//   
}


__global__ void multiply(CUDA_COMPLEX* crosscorr, CUDA_COMPLEX* density,CUDA_COMPLEX* refdensity, FLOAT* intensities, FLOAT sign, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        if(intensities[i]>=0){
                crosscorr[i].x = density[i].x*refdensity[i].x - (sign*density[i].y)*refdensity[i].y;
                crosscorr[i].y = density[i].x*refdensity[i].y + (sign*density[i].y)*refdensity[i].x;
            }
            else{
                crosscorr[i].x = 0;
                crosscorr[i].y = 0;
            }

    }
 
}




// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
    void GPU_context::get_shift(int& shiftx, int& shifty, FLOAT& shiftphase, bool& flip, FLOAT minr, FLOAT maxr){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);
        
        cudaMemcpyAsync(buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
        CUDA_COMPLEX density_sum;
        density_sum.x=0;
        density_sum.y=0;
        for(int i=0; i<npixels*npixels; i++){
            density_sum.x+=density_host[i].x;
            density_sum.y+=density_host[i].y;
            
        }

        FLOAT density_phase= atan2(density_sum.y,density_sum.x);
        FLOAT density_ref_phase= atan2(density_ref_sum.y,density_ref_sum.x);
                
        normalize_shift<<<gridSize, blockSize,0,main_stream>>>(density_dev, 1, npixels);
        

// // // // // // // // // // // // // // // 
       
        
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }        
        normalize_ft_shift<<<gridSize, blockSize,0,main_stream>>>(density_dev, 1, npixels);

// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
       
        
         //multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, -1, npixels);
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev, minr, maxr, -1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval1 = 0;
        int maxindex1 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval1){
                maxval1=val;
                maxindex1=i;
            }
        }
        
// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
        
     
//         multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, 1, npixels);
        
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev,minr,maxr, 1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval2 = 0;
        int maxindex2 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval2){
                maxval2=val;
                maxindex2=i;
            }
        }
// // // // // // // // // // // // // // // // // // // // // // // // // // // 
        
        if(maxval1>=maxval2){
            shiftx = maxindex1%npixels;
            shifty = maxindex1/npixels;
            shiftphase = density_ref_phase - density_phase;
            flip=false;            
        }
        
        else{
            shiftx = -maxindex2%npixels;
            shifty = -maxindex2/npixels;
            shiftphase = density_ref_phase + density_phase;
            flip=true;            
            }
        cudaMemcpyAsync(density_dev, buffer_dev_complex,  npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        cudaStreamSynchronize(main_stream);

    }


__global__ void shift(CUDA_COMPLEX* density, CUDA_COMPLEX* density_orig, SUPPORTTYPE* support, SUPPORTTYPE* support_orig,  int shiftx, int shifty, FLOAT shiftphase, bool flip, int npixels){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
            int oldindex=ix + iy*npixels;

            int newx = ix + shiftx;
            int newy = iy + shifty;
            if(newx<0) newx+=npixels;
            if(newx>=npixels) newx-=npixels;
            if(newy<0) newy+=npixels;
            if(newy>=npixels) newy-=npixels;            
            
            int newindex = newx + newy*npixels;
            if(flip)  newindex=npixels*npixels-1-newindex;
            FLOAT mod = std::sqrt(density_orig[oldindex].x*density_orig[oldindex].x+density_orig[oldindex].y*density_orig[oldindex].y);
            FLOAT phase = atan2(density_orig[oldindex].y,density_orig[oldindex].x);

            phase+= shiftphase;
            
            if(flip) phase=-phase;

            density[newindex].x = mod*std::cos(phase);
            density[newindex].y = mod*std::sin(phase);
            
            support[newindex] = support_orig[oldindex];
    
    }
    
}


 void GPU_context::reshift(int shiftx, int shifty, FLOAT shiftphase, bool flip){
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync(buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        cudaMemcpyAsync(buffer_dev, support_dev, npixels*npixels*sizeof(SUPPORTTYPE), 
                    cudaMemcpyDeviceToDevice, main_stream);             
        shift<<<gridSize, blockSize,0,main_stream>>>(density_dev, buffer_dev_complex, support_dev, buffer_dev, shiftx, shifty, shiftphase, flip, npixels);
        cudaStreamSynchronize(main_stream);


    }
    
    
    
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
    void GPU_context::get_symmetry(FLOAT& symmetry, FLOAT symexp,FLOAT symmin, FLOAT symmax){
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);
        
        cudaMemcpyAsync(buffer_dev_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
               
        normalize_shift<<<gridSize, blockSize,0,main_stream>>>(density_dev, 1, npixels);

        
        cudaMemcpyAsync( buffer_host_complex, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToHost, main_stream); 
        
        
        cudaStreamSynchronize(main_stream);
        
        
        if (cufftExecC2C(plan, density_dev, density_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }   
        
        normalize_ft_shift<<<gridSize, blockSize,0,main_stream>>>(density_dev, 1, npixels);

        
        cudaMemcpyAsync(density_ref_dev, density_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 

// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
       
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev, symmin, symmax, -1, npixels);

//         multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev,  -1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval1 = 0;
        int maxindex1 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval1){
                maxval1=val;
                maxindex1=i;
            }
        }
        
// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
        
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev, symmin, symmax, 1, npixels);

//         multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, density_dev, density_ref_dev, intensities_dev, 1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval2 = 0;
        int maxindex2 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval2){
                maxval2=val;
                maxindex2=i;
            }
        }
// // // // // // // // // // // // // // // // // // // // // // // // // // // 
        
        symmetry = std::sqrt(maxval2/maxval1);
        
        cudaMemcpyAsync(density_dev, buffer_dev_complex,  npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        cudaStreamSynchronize(main_stream);

    }

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // d


    void GPU_context::set_parents_density(COMPLEX* density1, COMPLEX* density2, COMPLEX* density3, COMPLEX* densityavg, COMPLEX* densitybest){
        cudaSetDevice(dev_id);
        memcpy(density1_host, density1, npixels*npixels*sizeof(CUDA_COMPLEX));
        memcpy(density2_host, density2, npixels*npixels*sizeof(CUDA_COMPLEX));
        memcpy(density3_host, density3, npixels*npixels*sizeof(CUDA_COMPLEX));
        memcpy(densityavg_host, densityavg, npixels*npixels*sizeof(CUDA_COMPLEX));
        memcpy(densitybest_host, densitybest, npixels*npixels*sizeof(CUDA_COMPLEX));

        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( density1_dev, density1_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream); 
        cudaMemcpyAsync( density2_dev, density2_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream); 
        cudaMemcpyAsync( density3_dev, density3_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);         
        cudaMemcpyAsync( densityavg_dev, densityavg_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);    
        cudaMemcpyAsync( densitybest_dev, densitybest_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                        cudaMemcpyHostToDevice, main_stream);                   
        cudaStreamSynchronize(main_stream);

    }


    void GPU_context::set_parents_support(SUPPORTTYPE* support1, SUPPORTTYPE* support2, SUPPORTTYPE* support3){
        cudaSetDevice(dev_id);
        memcpy(support1_host, support1, npixels*npixels*sizeof(SUPPORTTYPE));
        memcpy(support2_host, support2, npixels*npixels*sizeof(SUPPORTTYPE));
        memcpy(support3_host, support3, npixels*npixels*sizeof(SUPPORTTYPE));

        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync( support1_dev, support1_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);
        cudaMemcpyAsync( support2_dev, support2_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);
        cudaMemcpyAsync( support3_dev, support3_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);
        
        cudaStreamSynchronize(main_stream);

        
    }
    
    
    void GPU_context::get_parents_ft(){
            
        if (cufftExecC2C(plan, density1_dev, density1_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        if (cufftExecC2C(plan, density2_dev, density2_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        if (cufftExecC2C(plan, density3_dev, density3_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        if (cufftExecC2C(plan, densityavg_dev, densityavg_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
            
    }
    
    
// // // // // // // // // // // // // // // 


// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
    void GPU_context::reshift_parent(CUDA_COMPLEX* densitypar_host, CUDA_COMPLEX* densitypar_dev, SUPPORTTYPE* supportpar_dev, FLOAT minr, FLOAT maxr){
        
        int shiftx;
        int shifty;
        FLOAT shiftphase;
        bool flip;
        
        cudaSetDevice(dev_id);
        cudaStreamSynchronize(main_stream);
        
        cudaMemcpyAsync(buffer_dev_complex, densitypar_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        
        CUDA_COMPLEX density_sum;
        density_sum.x=0;
        density_sum.y=0;
        for(int i=0; i<npixels*npixels; i++){
            density_sum.x+=densitypar_host[i].x;
            density_sum.y+=densitypar_host[i].y;
            
        }

        FLOAT density_phase= atan2(density_sum.y,density_sum.x);
        FLOAT density_ref_phase= atan2(density_ref_sum.y,density_ref_sum.x);
                
        
        
        normalize_shift<<<gridSize, blockSize,0,main_stream>>>(densitypar_dev, 1, npixels);
        
        
        
        if (cufftExecC2C(plan, densitypar_dev, densitypar_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        }        

// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
       
        
         //multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, densitypar_dev, density_ref_dev, -1, npixels);
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, densitypar_dev, density_ref_dev, intensities_dev, minr, maxr, -1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval1 = 0;
        int maxindex1 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval1){
                maxval1=val;
                maxindex1=i;
            }
        }
        
// // // // // // // // // // // // // // // // // // // // // // // // // // // //         
        
     
//         multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, densitypar_dev, density_ref_dev, 1, npixels);
        
        multiply<<<gridSize, blockSize,0,main_stream>>>(crosscorr_dev, densitypar_dev, density_ref_dev, intensities_dev,minr,maxr, 1, npixels);

        if (cufftExecC2C(plan, crosscorr_dev, crosscorr_dev, CUFFT_FORWARD)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        } 
        
        cudaMemcpyAsync(crosscorr_host, crosscorr_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                cudaMemcpyDeviceToHost, main_stream); 
        cudaStreamSynchronize(main_stream);

        FLOAT maxval2 = 0;
        int maxindex2 = 0;
        
        for(int i=0; i<npixels*npixels; i++){
            FLOAT val = crosscorr_host[i].x*crosscorr_host[i].x+crosscorr_host[i].y*crosscorr_host[i].y;
            if(val>maxval2){
                maxval2=val;
                maxindex2=i;
            }
        }
// // // // // // // // // // // // // // // // // // // // // // // // // // // 
        
        if(maxval1>=maxval2){
            shiftx = maxindex1%npixels;
            shifty = maxindex1/npixels;
            shiftphase =density_ref_phase - density_phase;
            flip=false;            
        }
        
        else{
            shiftx = -maxindex2%npixels;
            shifty = -maxindex2/npixels;
            shiftphase = density_ref_phase + density_phase;
            flip=true;            
            }
        cudaMemcpyAsync(densitypar_dev, buffer_dev_complex,  npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        cudaStreamSynchronize(main_stream);
        
        
        
        
// // // // // // //  
        

        cudaMemcpyAsync(buffer_dev_complex, densitypar_dev, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyDeviceToDevice, main_stream); 
        cudaMemcpyAsync(buffer_dev, supportpar_dev, npixels*npixels*sizeof(SUPPORTTYPE), 
                    cudaMemcpyDeviceToDevice, main_stream);             
        shift<<<gridSize, blockSize,0,main_stream>>>(densitypar_dev, buffer_dev_complex, supportpar_dev, buffer_dev, shiftx, shifty, shiftphase, flip, npixels);
        
        cudaStreamSynchronize(main_stream);

    }



    
    
    
 void GPU_context::prepare_parents(FLOAT minr, FLOAT maxr){
        
    cudaSetDevice(dev_id);
    memcpy(density_ref_host, density1_host, npixels*npixels*sizeof(CUDA_COMPLEX));
        
    density_ref_sum.x=0;
    density_ref_sum.y=0;
    for(int i=0; i<npixels*npixels; i++){
        density_ref_sum.x+=density_ref_host[i].x;
        density_ref_sum.y+=density_ref_host[i].y;
    }
        
    cudaStreamSynchronize(main_stream);

    cudaMemcpyAsync( density_ref_dev, density_ref_host, npixels*npixels*sizeof(CUDA_COMPLEX), 
                    cudaMemcpyHostToDevice, main_stream);    
    
    normalize_shift<<<gridSize, blockSize,0,main_stream>>>(density_ref_dev,1., npixels);
    
    if (cufftExecC2C(plan, density_ref_dev, density_ref_dev, CUFFT_INVERSE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
    }

    cudaStreamSynchronize(main_stream);

     //////////
        
    reshift_parent(density_host, density_dev, support_dev, minr,maxr);
    reshift_parent(density2_host, density2_dev, support2_dev, minr,maxr);
    reshift_parent(density3_host, density3_dev, support3_dev, minr,maxr);
    reshift_parent(densityavg_host, densityavg_dev, buffer_dev, minr,maxr);
    reshift_parent(densitybest_host, densitybest_dev, buffer_dev, minr,maxr);

    }

    
    

__global__ void cross_density(CUDA_COMPLEX* density, 
                              CUDA_COMPLEX* density1,
                              CUDA_COMPLEX* density2,
                              CUDA_COMPLEX* density3,
                              CUDA_COMPLEX* densityavg,
                              CUDA_COMPLEX* densitybest,
                              SUPPORTTYPE* crossmap,
                              FLOAT* intensities,
                              FLOAT* intensities_avg,
                              int npixels, 
                              int crosssizex, int crosssizey, 
                              int squaresizex, int squaresizey, 
                              int offsetx, int offsety, FLOAT rotangle, FLOAT crossdiff, FLOAT use_average, bool crosssym
){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        int x = ix;
        int y = iy;
        
        if(x>=npixels/2) x-=npixels;
        if(y>=npixels/2) y-=npixels;
        
        FLOAT radius = std::sqrt(FLOAT(x*x+y*y)); 
        if(radius>= npixels/2.) radius = npixels/2-1;
        FLOAT angle = std::atan2(FLOAT(y),FLOAT(x));
        
        
        if(crosssym) if(angle<0) angle+=M_PI;
        
        angle+=rotangle;
        x = int(std::cos(angle)*radius);
        y = int(std::sin(angle)*radius);
        x+=npixels/2;
        y+=npixels/2;     
        
        int crossx= (x+offsetx)/squaresizex;
        int crossy= (y+offsety)/squaresizey;
        int crossindex = crossx+crossy*crosssizex;
        int crosscenterindex = (npixels/2+offsetx)/squaresizex +(npixels/2+offsety)/squaresizey*crosssizex;
        
        bool docross = (crossmap[crossindex]==1);
        
        if(docross){    
            
            
            CUDA_COMPLEX density1temp;
            density1temp.x = density1[i].x;
            density1temp.y = density1[i].y;
            
            FLOAT mod1 = std::sqrt(density1[i].x*density1[i].x + density1[i].y*density1[i].y);            
            
            if(intensities[i]<0){
                if(use_average>0){
                    FLOAT avgmod = (intensities_avg[i] + std::sqrt(densityavg[i].x*densityavg[i].x + densityavg[i].y*densityavg[i].y))*0.5;
                    FLOAT newmod = avgmod*(use_average) +(1.-use_average)*mod1;
                    if(mod1>0){
                        density1temp.x *= newmod/mod1;
                        density1temp.y *= newmod/mod1;
                    }                    
                }
/*                if(use_average<0){
                    FLOAT avgmod = intensities_avg[i];
                    FLOAT newmod = avgmod*std::abs(use_average) +(1.-std::abs(use_average))*mod1;
                    if(mod1>0){
                        density1temp.x *= newmod/mod1;
                        density1temp.y *= newmod/mod1;
                    }*/ 
                if(use_average<0){
                    if(mod1>0){
                        FLOAT locua = std::abs(use_average);
                        FLOAT maxmod = intensities_avg[i];
                        FLOAT minmod = std::sqrt(densityavg[i].x*densityavg[i].x + densityavg[i].y*densityavg[i].y);
                        FLOAT newmod = mod1;
                        if(newmod<minmod) newmod= minmod*locua + newmod*(1.-locua);
                        else if (newmod>maxmod) newmod=maxmod*locua + newmod*(1.-locua);
                        density1temp.x *= newmod/mod1;
                        density1temp.y *= newmod/mod1;
                    }
                }
            }
            
            
                        
            density[i].x = density1temp.x + crossdiff*(density2[i].x-density3[i].x);
            density[i].y = density1temp.y + crossdiff*(density2[i].y-density3[i].y);
                        
            
           
        }
        
        
        
    }
    
}


__global__ void cross_density_real(CUDA_COMPLEX* density, 
                              CUDA_COMPLEX* density1,
                              CUDA_COMPLEX* density2,
                              CUDA_COMPLEX* density3,
                              CUDA_COMPLEX* densityavg,
                              SUPPORTTYPE* crossmap,
                              FLOAT* intensities,
                              FLOAT* intensities_avg,
                              int npixels, 
                              int crosssizex, int crosssizey, 
                              int squaresizex, int squaresizey, 
                              int offsetx, int offsety, FLOAT rotangle, FLOAT crossdiff, FLOAT use_average, bool crosssym
){
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        int x = ix;
        int y = iy;
        
        if(x>=npixels/2) x-=npixels;
        if(y>=npixels/2) y-=npixels;
        
        FLOAT radius = std::sqrt(FLOAT(x*x+y*y)); 
        if(radius>= npixels/2.) radius = npixels/2-1;
        FLOAT angle = std::atan2(FLOAT(y),FLOAT(x));
        
        
        if(crosssym) if(angle<0) angle+=M_PI;
        
        angle+=rotangle;
        x = int(std::cos(angle)*radius);
        y = int(std::sin(angle)*radius);
        x+=npixels/2;
        y+=npixels/2;     
        
        int crossx= (x+offsetx)/squaresizex;
        int crossy= (y+offsety)/squaresizey;
        int crossindex = crossx+crossy*crosssizex;
        int crosscenterindex = (npixels/2+offsetx)/squaresizex +(npixels/2+offsety)/squaresizey*crosssizex;
        
        bool docrossphases = (crossmap[crossindex]==1);

        if(docrossphases){        

            density[i].x = density1[i].x + crossdiff*(density2[i].x-density3[i].x);
            density[i].y = density1[i].y + crossdiff*(density2[i].y-density3[i].y);
            
        }
        
        
    }
    
}


__global__ void cross_support(SUPPORTTYPE* support, 
                              SUPPORTTYPE* support1,
                              SUPPORTTYPE* support2,
                              SUPPORTTYPE* support3, 
                              int npixels){
    
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix<npixels && iy<npixels){
        int i = ix + iy*npixels;
        
//         support[i] = support[i] +support1[i]*support2[i]*support3[i];
//         support[i] = support[i] +support1[i];

//         support[i] = support1[i]+support2[i]-support3[i];
//         support[i] = support[i] + support1[i]+support2[i]*support3[i];
        
        
//         IMPLEMENT THE PAPER VERSION!!!
        
        support[i] = support1[i]+support2[i]*support3[i]*support[i];

        if(support[i]>0.5) support[i]=1;
        else support[i]=0;
    }
    
}

 void GPU_context::prepare_average(){
        cudaStreamSynchronize(main_stream);
       
        

        
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(density1_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum_1 = std::sqrt(error_host[0]);
                
        
        FLOAT newnorm = module_sum_1;

        
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(densityavg_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum_avg = std::sqrt(error_host[0]);

        // // // // // // // // // // // // //    
               
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_real_sum<<<gridSize, blockSize,0,main_stream>>>(intensities_avg_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum_avgint = std::sqrt(error_host[0])/FLOAT(npixels);
        
        
        // // // // // // // //         d

        
        normalize<<<gridSize, blockSize,0,main_stream>>>(densityavg_dev, newnorm/module_sum_avg , npixels);
        normalize<<<gridSize, blockSize,0,main_stream>>>(intensities_avg_dev, newnorm/module_sum_avgint , npixels);

        

 }


 void GPU_context::normalize_parents(FLOAT crossdiff){
        cudaStreamSynchronize(main_stream);
       
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(density1_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum1= std::sqrt(error_host[0]);
        
        // // // // // // // // // // // //  
        
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(densityavg_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum_avg = std::sqrt(error_host[0]);
        
        
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(density_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum = std::sqrt(error_host[0]);
        
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(density2_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum2 = std::sqrt(error_host[0]);
        
        
        // // // // // // // // // // // // // // // // // // // // // // // // // 
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_module_sum<<<gridSize, blockSize,0,main_stream>>>(density3_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum3 = std::sqrt(error_host[0]);
        
        
        // // // // // // // // // // // // //    
               
        cudaMemsetAsync(error_dev, 0, 2*sizeof(FLOAT), main_stream);    
        
        get_real_sum<<<gridSize, blockSize,0,main_stream>>>(intensities_avg_dev, error_dev, 2. , npixels); 

        cudaMemcpyAsync( error_host, error_dev, 2*sizeof(FLOAT), 
                        cudaMemcpyDeviceToHost, main_stream);  
        
        cudaStreamSynchronize(main_stream);

        FLOAT module_sum_avgint = std::sqrt(error_host[0])/FLOAT(npixels);
        
// // // // // // // // // //         d
        
        
        
        FLOAT newnorm = module_sum1*std::pow((module_sum2/module_sum3),crossdiff);
        
        cudaStreamSynchronize(main_stream);
        
        normalize<<<gridSize, blockSize,0,main_stream>>>(density_dev,    newnorm/module_sum , npixels);
        normalize<<<gridSize, blockSize,0,main_stream>>>(density1_dev,   newnorm/module_sum1 , npixels);        
        normalize<<<gridSize, blockSize,0,main_stream>>>(density2_dev,   newnorm/module_sum2 , npixels);
        normalize<<<gridSize, blockSize,0,main_stream>>>(density3_dev,   newnorm/module_sum3 , npixels);
        
        
        cudaStreamSynchronize(main_stream);
     
 }


    
 void GPU_context::get_new_individual(SUPPORTTYPE* crossmap, int crosssizex, int crosssizey, int squaresizex, int squaresizey, int offsetx, int offsety, FLOAT rotangle, FLOAT crossdiff, FLOAT use_average, bool crossft,bool crosssym, FLOAT real){
    
    memcpy(crossmap_host, crossmap, crosssizex*crosssizey*sizeof(SUPPORTTYPE));
    cudaMemcpyAsync( crossmap_dev, crossmap_host, crosssizex*crosssizey*sizeof(SUPPORTTYPE), 
                    cudaMemcpyHostToDevice, main_stream);

    
    apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density1_dev, support1_dev, npixels, real);
    apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, npixels, real);
    apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density2_dev, support2_dev, npixels, real);
    apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density3_dev, support3_dev, npixels, real);

    cudaStreamSynchronize(main_stream);
//     normalize_parents(crossdiff);
//     prepare_average();
                
    if(crossft){
        get_ft();
        get_parents_ft();                
    }
    
    cudaStreamSynchronize(main_stream);

    if(crossft)
        cross_density<<<gridSize, blockSize,0,main_stream>>>(density_dev,
                                                         density1_dev,
                                                         density2_dev,
                                                         density3_dev,
                                                         densityavg_dev,
                                                         densitybest_dev,
                                                         crossmap_dev,
                                                         intensities_dev,
                                                         intensities_avg_dev,
                                                         npixels, 
                                                         crosssizex, crosssizey, 
                                                         squaresizex, squaresizey,
                                                         offsetx, offsety, rotangle, crossdiff, use_average, crosssym);
        
    else     
        cross_density_real<<<gridSize, blockSize,0,main_stream>>>(density_dev,
                                                         density1_dev,
                                                         density2_dev,
                                                         density3_dev,
                                                         densityavg_dev,
                                                         crossmap_dev,
                                                         intensities_dev,
                                                         intensities_avg_dev,
                                                         npixels, 
                                                         crosssizex, crosssizey, 
                                                         squaresizex, squaresizey,
                                                         offsetx, offsety, rotangle, crossdiff, use_average, crosssym);

    cross_support<<<gridSize, blockSize,0,main_stream>>>(support_dev,
                                                         support1_dev,
                                                         support2_dev,
                                                         support3_dev,
                                                         npixels);    
      
    cudaStreamSynchronize(main_stream);

//     sub_intensities<<<gridSize, blockSize,0,main_stream>>>(density_dev, intensities_dev,  npixels); 
    
    if(crossft){
        get_ift();        
    }
    
//     normalize_parents(crossdiff);

    
    
    apply_support_er<<<gridSize, blockSize,0,main_stream>>>(density_dev, support_dev, npixels, real);


}
    
    
    
void GPU_context::set_avg_intensities(FLOAT* intensities_avg){
        cudaSetDevice(dev_id);
        memcpy(intensities_avg_host, intensities_avg, npixels*npixels*sizeof(SUPPORTTYPE));
        cudaStreamSynchronize(main_stream);

        cudaMemcpyAsync(intensities_avg_dev, intensities_avg_host, npixels*npixels*sizeof(SUPPORTTYPE), 
                        cudaMemcpyHostToDevice, main_stream);
        cudaStreamSynchronize(main_stream);

        
    }

    
    
    
// // // // // // // // // // // d


 void GPU_context::add_module_to_buffer(COMPLEX* data, bool ft){
     
        set_density(data);
        
        if(ft) get_ft();
        
        cudaStreamSynchronize(main_stream);          
        add_module<<<gridSize, blockSize,0,main_stream>>>(density_dev, buffer_dev_float, npixels);
        cudaStreamSynchronize(main_stream);

        if(ft) get_ift();

    }


 void GPU_context::add_to_buffer(COMPLEX* data){
     
        set_density(data);
        cudaStreamSynchronize(main_stream);          
        add<<<gridSize, blockSize,0,main_stream>>>(density_dev, buffer_dev_complex, npixels);
        cudaStreamSynchronize(main_stream);


    }



 void GPU_context::add_to_buffer(SUPPORTTYPE* data){
     
        set_support(data);
        cudaStreamSynchronize(main_stream);          
        add<<<gridSize, blockSize,0,main_stream>>>(support_dev, buffer_dev, npixels);
        cudaStreamSynchronize(main_stream);


    }
    
