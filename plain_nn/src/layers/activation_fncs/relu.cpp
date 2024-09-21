#include "activation_fncs.h"

ReLU::ReLU(){
    this->fn_type = ActivationType::RELU;
}


double ReLU::forward(const double input){
    return std::max(0.0, input);
}


double ReLU::backward(const double input){
    return input > 0 ? 1 : 0;
}