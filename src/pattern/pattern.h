#pragma once

#include <vector>
#include <array>
#include <tuple>


class Pattern{
public:
    Pattern(double* data_in, std::array<std::size_t,2> dims_in, std::array<double,2> center_in, double dtheta_in):
            data(data_in), dims(dims_in), center(center_in), dtheta(dtheta_in){       
//             printf("%d,%d\n\n", dims[0], dims[1]);
//             printf("%f,%f\n\n", center[0], center[1]);

    };


    std::vector<double> downscale(std::array<std::size_t,2> newdims);
    std::vector<double> downscale_and_correct(double R, double dx, std::array<std::size_t,2> newdims, std::vector<double> mask);

    double get_data(std::array<std::size_t,2> coords);
    
    double* data;
    std::array<std::size_t,2>  dims;
    std::array<double,2> center;
    double dtheta;
    std::tuple<std::vector<double>,std::vector<double>> cut_downscale_and_correct(double R, double dx, std::array<std::size_t,2> cutdims, std::array<std::size_t,2> newdims, std::vector<double> mask);
    std::tuple<std::vector<double>,std::vector<double>> cut_and_downscale(std::array<std::size_t,2> cutdims, std::array<std::size_t,2> newdims, std::vector<double> mask);
    
};
