#include "activation_fncs.hpp"
#include "tensor.hpp"
#include <algorithm>

ReLU::ReLU(){
    this->fn_type = ActivationType::RELU;
}


Tensor ReLU::forward(Tensor& input){
    std::vector<double> output;
    
    double *_input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(std::max(0.0, x));});

    return Tensor(input.shape(), output);
}


Tensor ReLU::backward(Tensor& input){

    std::vector<double> output;

    double *_input = input.data();
    std::for_each(_input, _input+input.size(), [&output](double& x){output.push_back(x > 0 ? 1 : 0);});

    return Tensor(input.shape(), output);
}