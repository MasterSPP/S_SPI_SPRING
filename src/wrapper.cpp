#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <variant>
#include <string>
#include <tuple>
#include <omp.h>
#include <tuple>

#include "reconstruction.h"
#include "datatypes.h"
#include "solver.h"
#include "cpu/cpu_context.h"
#include "pattern/pattern.h"

#ifdef WITH_CUDA
#include "gpu/gpu_utils.h"
#endif


namespace py = pybind11;


PYBIND11_MODULE(spring_core, sp)
{
    
    py::module pr = sp.def_submodule("phase_retrieval");

    
    
    auto PyReconstruction = py::class_<Reconstruction, std::shared_ptr<Reconstruction>>(pr, "Reconstruction")
                .def(py::init([]( py::array_t<std::complex<FLOAT>> density, py::array_t<SUPPORTTYPE> support) {

                auto densitybuf = density.request();
                auto supportbuf = support.request();
                
                return new Reconstruction((COMPLEX*) densitybuf.ptr, (SUPPORTTYPE*) supportbuf.ptr, std::array<std::size_t,2>({densitybuf.shape[0],densitybuf.shape[1]})); 
                }),R"pbdoc(
                            define the reconstruction object
                            )pbdoc", 
                    py::arg("density"),py::arg("support"))
                .def("get_density", [](Reconstruction &rec){
                    auto density = rec.get_density();
                    auto nparray = py::array(density.size(), density.data());
                    nparray.resize({rec.dims[0], rec.dims[1]});
                    return nparray;
                })
                .def("get_support", [](Reconstruction &rec){
                    auto support = rec.get_support();
                    auto nparray = py::array(support.size(), support.data());
                    nparray.resize({rec.dims[0], rec.dims[1]});
                    return nparray;
                })
                .def("set_support", [](Reconstruction &rec, py::array_t<SUPPORTTYPE> support){
                    auto supportbuf = support.request();
                    rec.set_support((SUPPORTTYPE*) supportbuf.ptr);
                })
                .def("set_density", [](Reconstruction &rec, py::array_t<std::complex<FLOAT>> density){
                    auto densitybuf = density.request();
                    rec.set_density((COMPLEX*) densitybuf.ptr);
                })
                .def("get_error", [](Reconstruction &rec){
                    return rec.get_error();
                })
                .def("get_real_error", [](Reconstruction &rec){
                    return rec.get_real_error();
                })
                .def("get_fourier_error", [](Reconstruction &rec){
                    return rec.get_fourier_error();
                })
                .def("get_reality", [](Reconstruction &rec){
                    return rec.get_reality();
                })
                .def("get_symmetry", [](Reconstruction &rec){
                    return rec.get_symmetry();
                })
                .def("get_module_sum", [](Reconstruction &rec){
                    return rec.get_module_sum();
                })
                .def("get_complex_sum", [](Reconstruction &rec){
                    return rec.get_complex_sum();
                }); 
    
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
    auto PySolver = py::class_<Solver, std::shared_ptr<Solver>>(pr, "Solver")
                .def(py::init([]( py::array_t<FLOAT> pattern, std::array<int,2> center_in,  FLOAT reality, FLOAT realpen, int nthreads, bool use_bounds, std::vector<int> gpu_ids, int seed) {

                auto patternbuf = pattern.request();
                
                return new Solver((FLOAT*) patternbuf.ptr, std::array<std::size_t,2>({patternbuf.shape[0],patternbuf.shape[1]}), center_in, reality, realpen, nthreads,use_bounds, gpu_ids,seed); 
                }),R"pbdoc(
                            define the solver object
                            )pbdoc", 
                    py::arg("pattern"),py::arg("center")= std::array<int,2>({-1,-1}), py::arg("reality")=0, py::arg("realpen")=0, py::arg("nthreads")=0, py::arg("use_bounds")=true, py::arg("gpus")= std::vector<int>({}), py::arg("seed")= 17)
                .def("get_pattern", [](Solver &sol){
                    auto pattern = sol.get_pattern();
                    auto nparray = py::array(pattern.size(), pattern.data());
                    nparray.resize({sol.dims[0], sol.dims[1]});
                    return nparray;
                })
                .def("set_pattern", [](Solver &sol, py::array_t<FLOAT> pattern){
                    auto patternbuf = pattern.request();
                    sol.set_pattern((FLOAT*) patternbuf.ptr);
                })
                .def("set_shift_param", [](Solver &sol, FLOAT minr_, FLOAT maxr_){
                    sol.set_shift_param(minr_, maxr_);
                },py::arg("shiftmin")=0., py::arg("shiftmax")=2.)
                .def("Sequence", [](Solver &sol,
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec, int nrep)
                                    -> 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        >
                                    {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.Sequence(rec_array, nrep);
                        return result[0];
                    }
                    else{
                        return sol.Sequence(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec), nrep);
                    }
//                     }, py::arg("reconstruction"), py::arg("nrep")=1)
                }, py::arg("reconstruction"), py::arg("nrep")=1, py::call_guard<py::gil_scoped_release>())
                .def("Reshift", [](Solver &sol,
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec,
                                    std::shared_ptr<Reconstruction> ref
                                  )
                            -> std::variant<std::shared_ptr<Reconstruction>,std::vector<std::shared_ptr<Reconstruction>>>
                                    {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.Reshift(rec_array,ref);
                        return result[0];
                    }
                    else{
                        return sol.Reshift(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec), ref);
                    }
                }, py::arg("reconstruction"), py::arg("reference"))
                .def("Crossover", [](Solver &sol, std::vector<std::shared_ptr<Reconstruction>> rec, std::shared_ptr<Reconstruction> bestrec,std::shared_ptr<Reconstruction> avgrec, FLOAT crossbalance, FLOAT crossdiff, FLOAT crossexp, FLOAT crossft, bool crosssym,FLOAT use_best, bool update_support)
                                    {
                                    return sol.Crossover(rec, bestrec,avgrec, crossbalance, crossdiff, crossexp, crossft, crosssym, use_best, update_support);
                                    }
                                    , py::arg("reconstruction"), py::arg("best"), py::arg("average"),py::arg("crossbalance"), py::arg("crossdiff"), py::arg("crossexp")=1, py::arg("crossft")=true, py::arg("crosssym")=false,  py::arg("usebest")=0, py::arg("update_support")=true, py::call_guard<py::gil_scoped_release>())
                
                .def("HIO", [](Solver &sol, 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec,int iterations, FLOAT beta)
                        -> std::variant<std::shared_ptr<Reconstruction>,std::vector<std::shared_ptr<Reconstruction>>>
                {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.HIO(rec_array, iterations,beta);
                        return result[0];
                    }
                    else{
                        return sol.HIO(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec),iterations,beta);
                    }  
                }, py::arg("reconstruction"), py::arg("iterations"), py::arg("beta"))
                .def("RAAR", [](Solver &sol, 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec,int iterations, FLOAT beta)
                        -> std::variant<std::shared_ptr<Reconstruction>,std::vector<std::shared_ptr<Reconstruction>>>
                {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.RAAR(rec_array, iterations,beta);
                        return result[0];
                    }
                    else{
                        return sol.RAAR(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec),iterations,beta);
                    }  
                }, py::arg("reconstruction"), py::arg("iterations"), py::arg("beta"))
                .def("ER", [](Solver &sol, 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec, int iterations)
                    -> std::variant<std::shared_ptr<Reconstruction>,std::vector<std::shared_ptr<Reconstruction>>>
                {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.ER(rec_array, iterations);
                        return result[0];
                    }
                    else{
                        return sol.ER(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec),iterations);
                    }                    
                }, py::arg("reconstruction"), py::arg("iterations"))
                .def("SW", [](Solver &sol,
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec, FLOAT sigma, FLOAT threshold)
                                    -> 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        >
                                    {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.SW(rec_array, sigma, threshold);
                        return result[0];
                    }
                    else{
                        return sol.SW(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec), sigma, threshold);
                    }
                }, py::arg("reconstruction"),  py::arg("sigma"), py::arg("threshold"))
                .def("SWA", [](Solver &sol,
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec, FLOAT sigma, int pixels)
                                    -> 
                                    std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        >
                                    {
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);
                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        auto result=sol.SWA(rec_array, sigma, pixels);
                        return result[0];
                    }
                    else{
                        return sol.SWA(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec), sigma, pixels);
                    }
                }, py::arg("reconstruction"),  py::arg("sigma"), py::arg("pixels"))               
                
                .def("Evaluate", [](Solver &sol, std::variant<
                                        std::shared_ptr<Reconstruction>,
                                        std::vector<std::shared_ptr<Reconstruction>>
                                        > rec){
                    auto* rec_ptr  = std::get_if<std::shared_ptr<Reconstruction>>(&rec);

                    if(rec_ptr!=nullptr){
                        std::vector<std::shared_ptr<Reconstruction>> rec_array(1);
                        rec_array[0] = *rec_ptr;
                        sol.Evaluate(rec_array);
                    }
                    else{
                        sol.Evaluate(std::get<std::vector<std::shared_ptr<Reconstruction>>>(rec));
                    }
                })
                .def("set_sequence", [](Solver &sol, std::vector<std::tuple<std::string, std::vector<FLOAT>>> sequence){
                    sol.set_sequence(sequence);
                })
                .def("set_reality", [](Solver &sol, FLOAT reality){
                    sol.set_reality(reality);
                })
// // // //                 
                .def("InitializePop", [](Solver &sol, int nrec, int support_size, std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling, FLOAT phase,  FLOAT suppcut)
                                        -> std::vector<std::shared_ptr<Reconstruction>>
                                    {
                                        return sol.Initialize_pop(nrec, support_size, itemsize, nitems,  exponent, scaling, phase,  suppcut);
                                    }, py::arg("popsize"),  py::arg("suppsize"), py::arg("itemsize"), py::arg("nitems")=8,  py::arg("exponent")=0, py::arg("scaling")=0, py::arg("phaserange")=0, py::arg("suppcut")=0)
                

                .def("InitializePopRef", [](Solver &sol, int nrec, std::shared_ptr<Reconstruction> refrec,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling, FLOAT phase,  FLOAT suppcut)
                                    -> std::vector<std::shared_ptr<Reconstruction>>
                                    {
                                        return sol.Initialize_pop(nrec, refrec, itemsize, nitems,  exponent, scaling, phase,  suppcut);
                                    }, py::arg("popsize"),  py::arg("reference"), py::arg("itemsize"), py::arg("nitems")=8,  py::arg("exponent")=0, py::arg("scaling")=0, py::arg("phaserange")=0, py::arg("suppcut")=0)
                
                .def("InitializePopSupp", [](Solver &sol, int nrec,  py::array_t<SUPPORTTYPE> refsupport,  FLOAT initfrac, std::array<FLOAT,2> scaling_in, FLOAT phase_in)
                                        -> std::vector<std::shared_ptr<Reconstruction>>
                                    {
                                        auto supportbuf = refsupport.request();
                                        std::vector<SUPPORTTYPE> refsupp(sol.dims[0]*sol.dims[1]);
                                        memcpy(&refsupp[0], (SUPPORTTYPE*) supportbuf.ptr, sol.dims[0]*sol.dims[1]*sizeof(SUPPORTTYPE));     
                                        return sol.Initialize_pop( nrec, refsupp,  initfrac,scaling_in,  phase_in);
                                    }, py::arg("popsize"),  py::arg("reference"), py::arg("fraction"), py::arg("scaling"), py::arg("phaserange"))                
                
                
                .def("GetAverage", [](Solver &sol,  std::vector<std::shared_ptr<Reconstruction>> rec, FLOAT fraction)
                                    -> std::shared_ptr<Reconstruction>{
                                    return sol.GetAverage(rec, fraction);
                                    }, py::arg("reconstruction"), py::arg("fraction")=1.)
                
                .def("GetAverageModule", [](Solver &sol,  std::vector<std::shared_ptr<Reconstruction>> rec, bool ft, FLOAT fraction){
                                    auto avgmod =  sol.GetAverageModule(rec, ft, fraction);
                                    auto nparray = py::array(avgmod.size(), avgmod.data());
                                    nparray.resize({sol.dims[0], sol.dims[1]});
                                    return nparray;
                                    }, py::arg("reconstruction"), py::arg("ft")=true, py::arg("fraction")=1.)
                ;
                

                
// // // // // // // // // // // // // // //         
                
        py::module pm = sp.def_submodule("pattern");

        pm.def("downscale", [](py::array_t<double> data_in,  std::array<std::size_t,2> newsize){
                        Pattern pattern((double*) (std::size_t) data_in.request().ptr , std::array<std::size_t,2>{(std::size_t) data_in.request().shape[1], (std::size_t) data_in.request().shape[0]}, std::array<double,2>{0,0}, 1);
                               
                        auto result=pattern.downscale(newsize);
                        
                        auto nparray = py::array(result.size(), result.data());
                        nparray.resize({newsize[1],newsize[0]});
                        return nparray;

                        },R"pbdoc(
                            Downscale the given pattern to the given size. Note that negative values in the input data are ignored.
                            
                            :param data: 2d numpy array, containing the diffraction image
                            :param newsize: 2-element array with the new size
                            
                            :returns: the rescaled diffraction image
                            )pbdoc", py::return_value_policy::move,  py::arg("data"), py::arg("newsize"));
         
         pm.def("downscale_and_correct", [](py::array_t<double> data_in, py::array_t<double> mask_in, std::array<double,2> center_in, double R, double dx, std::array<std::size_t,2> newsize){
                        Pattern pattern((double*) (std::size_t) data_in.request().ptr , std::array<std::size_t,2>{(std::size_t) data_in.request().shape[1], (std::size_t) data_in.request().shape[0]}, center_in, 1);
                               
                        double* mask_array = (double*) mask_in.request().ptr;
                        std::vector<double> mask(data_in.request().shape[0]*data_in.request().shape[1]);
                        for(int i=0; i<data_in.request().shape[0]*data_in.request().shape[1];i++){
                            mask[i]=mask_array[i];
                        }
                        auto result=pattern.downscale_and_correct(R, dx,newsize,mask);
                        
                        auto nparray = py::array(result.size(), result.data());
                        nparray.resize({newsize[1],newsize[0]});
                        return nparray;

                        },R"pbdoc(
                            Downscale the given pattern to the given size. Note that negative values in the input data are ignored.
                            
                            :param data: 2d numpy array, containing the diffraction image
                            :param mask: 2d numpy array, containing the mask
                            :param center: 2-elements array containing the x and y coordinate of the pattern center
                            :param distance: detector distance
                            :param pixel_size: pixel size
                            :param newsize: 2-element array with the new size
                            
                            :returns: the rescaled diffraction image
                            )pbdoc", py::return_value_policy::move,  py::arg("data"), py::arg("mask"), py::arg("center"), py::arg("distance"), py::arg("pixel_size"),  py::arg("newsize")); 
                
         pm.def("cut_downscale_and_correct", [](py::array_t<double> data_in, py::array_t<double> mask_in, std::array<double,2> center_in, double R, double dx, std::array<std::size_t,2> cutsize, std::array<std::size_t,2> newsize){
                        Pattern pattern((double*) (std::size_t) data_in.request().ptr , std::array<std::size_t,2>{(std::size_t) data_in.request().shape[1], (std::size_t) data_in.request().shape[0]}, center_in, 1);
                               
                        double* mask_array = (double*) mask_in.request().ptr;
                        std::vector<double> mask(data_in.request().shape[0]*data_in.request().shape[1]);
                        for(int i=0; i<data_in.request().shape[0]*data_in.request().shape[1];i++){
                            mask[i]=mask_array[i];
                        }
                        auto result=pattern.cut_downscale_and_correct(R, dx,cutsize, newsize,mask);
                        
                        auto newpattern = std::get<0>(result);
                        auto newmask = std::get<1>(result);
                        
                        
                        auto nparray_pattern = py::array(newpattern.size(), newpattern.data());
                        nparray_pattern.resize({newsize[1],newsize[0]});
                        
                        auto nparray_mask = py::array(newmask.size(), newmask.data());
                        nparray_mask.resize({newsize[1],newsize[0]});
                        
                        return std::make_tuple(nparray_pattern,nparray_mask);

                        },R"pbdoc(
                            Downscale the given pattern to the given size. Note that negative values in the input data are ignored.
                            
                            :param data: 2d numpy array, containing the diffraction image
                            :param mask: 2d numpy array, containing the mask
                            :param center: 2-elements array containing the x and y coordinate of the pattern center
                            :param distance: detector distance
                            :param pixel_size: pixel size
                            :param newsize: 2-element array with the new size
                            
                            :returns: the rescaled diffraction image
                            )pbdoc", py::return_value_policy::move,  py::arg("data"), py::arg("mask"), py::arg("center"), py::arg("distance"), py::arg("pixel_size"),  py::arg("cutsize"),  py::arg("newsize"));     
         pm.def("cut_and_downscale", [](py::array_t<double> data_in, py::array_t<double> mask_in, std::array<double,2> center_in, std::array<std::size_t,2> cutsize, std::array<std::size_t,2> newsize){
                        Pattern pattern((double*) (std::size_t) data_in.request().ptr , std::array<std::size_t,2>{(std::size_t) data_in.request().shape[1], (std::size_t) data_in.request().shape[0]}, center_in, 1);
                               
                        double* mask_array = (double*) mask_in.request().ptr;
                        std::vector<double> mask(data_in.request().shape[0]*data_in.request().shape[1]);
                        for(int i=0; i<data_in.request().shape[0]*data_in.request().shape[1];i++){
                            mask[i]=mask_array[i];
                        }
                        auto result=pattern.cut_and_downscale(cutsize, newsize,mask);
                        
                        auto newpattern = std::get<0>(result);
                        auto newmask = std::get<1>(result);
                        
                        
                        auto nparray_pattern = py::array(newpattern.size(), newpattern.data());
                        nparray_pattern.resize({newsize[1],newsize[0]});
                        
                        auto nparray_mask = py::array(newmask.size(), newmask.data());
                        nparray_mask.resize({newsize[1],newsize[0]});
                        
                        return std::make_tuple(nparray_pattern,nparray_mask);

                        },R"pbdoc(
                            Downscale the given pattern to the given size. Note that negative values in the input data are ignored.
                            
                            :param data: 2d numpy array, containing the diffraction image
                            :param mask: 2d numpy array, containing the mask
                            :param center: 2-elements array containing the x and y coordinate of the pattern center
                            :param newsize: 2-element array with the new size
                            
                            :returns: the rescaled diffraction image
                            )pbdoc", py::return_value_policy::move,  py::arg("data"), py::arg("mask"), py::arg("center"),  py::arg("cutsize"),  py::arg("newsize"));     




                
// // // // // // // // // // // // // // //         
             
#ifdef WITH_CUDA

        py::module gu = sp.def_submodule("GPUtils");

        gu.def("get_count", [](){
                        GPU_utils gputils;
                        return gputils.get_count();
                        },R"pbdoc(
                            Get the number of available CUDA-capable GPUs.
                            )pbdoc");
        gu.def("get_ids", [](){
                        GPU_utils gputils;
                        return gputils.get_ids();
                        },R"pbdoc(
                            Get the list of available CUDA-capable GPUs.
                            )pbdoc");
        gu.def("get_usage", [](){
                        GPU_utils gputils;
                        return gputils.get_usage();
                        },R"pbdoc(
                            Get the usage of available CUDA-capable GPUs.
                            )pbdoc");
        
        gu.def("get_info", [](){
                GPU_utils gputils;
                return gputils.get_info();
                },R"pbdoc(
                    Get the info of available CUDA-capable GPUs.
                    )pbdoc");
        
#else
        py::module gu = sp.def_submodule("GPUtils");

        gu.def("get_count", [](){
                            return 0;
                        },R"pbdoc(
                            Get the number of available CUDA-capable GPUs.
                            )pbdoc");
        gu.def("get_ids", [](){
                            return std::vector<int>(0);
                        },R"pbdoc(
                            Get the list of available CUDA-capable GPUs.
                            )pbdoc");
        gu.def("get_usage", [](){
                            return std::vector<int>(0);
                        },R"pbdoc(
                            Get the usage of available CUDA-capable GPUs.
                            )pbdoc");
        
        gu.def("get_info", [](){
                            return std::vector<std::string>(0) ;
                        },R"pbdoc(
                            Get the info of available CUDA-capable GPUs.
                            )pbdoc");
         
#endif
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 


}
