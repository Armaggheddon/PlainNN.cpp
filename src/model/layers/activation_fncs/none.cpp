#include "activation_fncs.h"

None::None(){
    this->fn_type = ActivationType::NONE;
}

double None::forward(const double input){
    return input;
}

double None::backward(const double input){
    return 1;
}