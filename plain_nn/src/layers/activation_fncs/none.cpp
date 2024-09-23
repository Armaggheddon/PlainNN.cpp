#include "activation_fncs.hpp"
#include "tensor.hpp"

None::None(){
    this->fn_type = ActivationType::NONE;
}

Tensor None::forward(Tensor& input){
    return input;
}

Tensor None::backward(Tensor& input){
    return Tensor(input.shape(), false, 1.0);
}