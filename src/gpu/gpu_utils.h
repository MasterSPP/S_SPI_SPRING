#pragma once
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <memory>
#include <vector>
#include <string>


class GPU_utils{
public:
    GPU_utils();
    int get_count();
    std::vector<int> get_ids();
    std::vector<std::string> get_info();
    std::vector<int> get_usage();
};
