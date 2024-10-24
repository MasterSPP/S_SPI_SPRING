cmake_minimum_required(VERSION 3.17)
include(CheckLanguage)
cmake_policy(SET CMP0091 NEW)

set(PYBIND11_PYTHON_VERSION 3)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# # # # # # # # # # # # # # # # # # # # # 

option(SINGLE_PRECISION "Use single precision" ON)
option(WITH_OMP "Use OpenMP" ON)
option(WITH_CUDA "Use CUDA" ON)
option(WITH_NVML "Use NVML" ON)


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -fopenmp-simd")

set(CMAKE_POSITION_INDEPENDENT_CODE True)



if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()


# # # # # #  CHECK CUDA

if(WITH_CUDA)
    project(spring_core VERSION 1.0.0 LANGUAGES CUDA CXX C)
    check_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    
    if(WITH_NVML)
        message(STATUS "NVML support activated")
    else()
        message(STATUS "No NVML support")
    endif()
else()
  message(STATUS "No CUDA support")
  set(WITH_CUDA OFF)
  project(spring_core VERSION 1.0.0 LANGUAGES CXX C)
endif()



  
# # # # # # #  FETCHING PYBIND11 

include(FetchContent)
FetchContent_Declare(
    pybind11 
    GIT_REPOSITORY https://github.com/pybind/pybind11
    #GIT_TAG        v2.4.3
    GIT_TAG        v2.12.1
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(spring_core ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp)
target_include_directories(spring_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp PROPERTY CXX_STANDARD 17)

set(PYBIND11_CPP_STANDARD -std=c++17)

# # # # # # # # # # # # # # # # # # # # # # # # # 

if(WITH_OMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    if(WITH_CUDA AND NOT WIN32)
         set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler='-fopenmp' " )   
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()




 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


add_library(pattern STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/pattern/pattern.cpp)
target_include_directories(pattern PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/pattern/)


if(WITH_CUDA)
    message(STATUS "CUDA compiler: " ${CMAKE_CUDA_COMPILER_ID} "  Version: " ${CMAKE_CUDA_COMPILER_VERSION})

    
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if(WITH_NVML)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --default-stream per-thread --device-link -DWITH_NVCC -DWITH_NVML") 
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --default-stream per-thread --device-link -DWITH_NVCC") 
    endif()


  
    add_definitions(-DWITH_CUDA)
    add_library(gpu_context STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_context.cu)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_context.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(gpu_context PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(gpu_context PRIVATE CUDA::cufft)
    target_link_libraries(gpu_context PRIVATE CUDA::cudart)

    
    add_library(solver STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(solver PRIVATE gpu_context)

    
    add_library(gpu_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_utils.cu)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_utils.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(gpu_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(gpu_utils PRIVATE CUDA::cudart)

    if(WITH_NVML)    
        target_link_libraries(gpu_utils PRIVATE CUDA::nvml)
    endif()
    
    target_link_libraries(spring_core PRIVATE gpu_utils)

    

    
else()
    add_library(solver STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cpp)
    target_include_directories(solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    
endif()


target_link_libraries(spring_core PRIVATE solver)
target_link_libraries(spring_core PRIVATE pattern)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # d

install(TARGETS spring_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


