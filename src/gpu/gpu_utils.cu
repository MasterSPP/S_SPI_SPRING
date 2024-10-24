    
#include "gpu/gpu_utils.h"
#include <iostream>

#ifdef WITH_NVML
    #include <nvml.h>
#endif

GPU_utils::GPU_utils(){      
}

       
int GPU_utils::get_count(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
            
}

std::vector<int> GPU_utils::get_ids(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::vector<int> ids;
    for(int i=0; i<deviceCount; i++){
        ids.push_back(i);
    }
    
    return ids;
            
}

std::vector<std::string> GPU_utils::get_info(){
    cudaDeviceProp deviceProp;
    
    std::vector<std::string> info;
    
    int ndev = get_count();
    
    for(int id = 0; id<ndev; id++){    
        cudaGetDeviceProperties(&deviceProp, id);
        std::string name(deviceProp.name);
        info.push_back(name);
    }       
    return info;
//     printf("Device %d has compute capability %d.%d.\n",
//            id, deviceProp.major, deviceProp.minor);
}


#ifdef WITH_NVML

std::vector<int> GPU_utils::get_usage(){
    nvmlReturn_t result;
    unsigned int deviceCount;
    std::vector<int> usage;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
//         return -1;
    }

    // Get the number of GPUs
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
//         return -1;
    }

    for (unsigned int i = 0; i < deviceCount; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
            continue;
        }

        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get utilization rates: " << nvmlErrorString(result) << std::endl;
            continue;
        }

        usage.push_back(utilization.gpu);
    }

    // Shutdown NVML
    nvmlShutdown();
    return usage;
    
}

#else 

std::vector<int> GPU_utils::get_usage(){
    
//     std::cerr << "WARNING: Spring built without NVIDIA Management Library support. GPU priority follows GPU id. Compile with WITH_NVML=On for NVML support." << std::endl;

    std::vector<int> usage;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0; i<deviceCount; i++){
        usage.push_back(-1);
    }
    
    return usage;    
}

#endif



