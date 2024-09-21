#include "activation_fncs.hpp"

None::None(){
    this->fn_type = ActivationType::NONE;
}

double None::forward(const double input){
    return input;
}

double None::backward( __attribute_maybe_unused__ const double input){
    return 1;
}