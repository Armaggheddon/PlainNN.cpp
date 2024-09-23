#include "activation_fncs.hpp"
#include "tensor.hpp"

#include <cmath>
#include <algorithm>

Softmax::Softmax(){
    this->fn_type = ActivationType::SOFTMAX;
}

Tensor Softmax::forward(Tensor& input){
    std::vector<double> output(input.size());
    double *_input = input.data();
    double max = *std::max_element(_input, _input + input.size());
    double sum = 0.0;
    for (int i = 0; i < input.size(); i++){
        output[i] = std::exp(_input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < input.size(); i++){
        output[i] /= sum;
    }
    return Tensor(input.shape(), output);
}

Tensor Softmax::backward(Tensor& input){
    std::vector<double> output(input.size());
    double *_input = input.data();
    for (int i = 0; i < input.size(); i++){
        output[i] = _input[i] * (1 - _input[i]);
    }
    return Tensor(input.shape(), output);
}