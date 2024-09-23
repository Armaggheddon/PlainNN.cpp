#include "activation_fncs.hpp"
#include "tensor.hpp"
#include <cmath>
#include <algorithm>

Tanh::Tanh(){
    this->fn_type = ActivationType::TANH;
}

Tensor Tanh::forward(Tensor& input){

    std::vector<double> output;

    double* _input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(std::tanh(x));});

    return Tensor(input.shape(), output);
}

Tensor Tanh::backward(Tensor& input){

    std::vector<double> output;

    double* _input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(1 - std::pow(std::tanh(x), 2));});
    
    return Tensor(input.shape(), output);
}