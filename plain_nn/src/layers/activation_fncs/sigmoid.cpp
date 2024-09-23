#include "activation_fncs.hpp"
#include "tensor.hpp"
#include <cmath>
#include <algorithm>

Sigmoid::Sigmoid(){
    this->fn_type = ActivationType::SIGMOID;
}

Tensor Sigmoid::forward(Tensor& input){

    std::vector<double> output;

    double* _input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(1 / (1 + std::exp(-x)));});

    return Tensor(input.shape(), output);
}

Tensor Sigmoid::backward(Tensor& input){

    std::vector<double> output;

    double* _input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(x * (1 - x));});

    return Tensor(input.shape(), output);
}