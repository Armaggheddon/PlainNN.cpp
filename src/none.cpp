#include <vector>
#include "activation.h"

None::None(){
    this->type = ActivationFnType::NONE;
}

void None::forward(std::vector<float> *matrix){
    return;
}

void None::backward(std::vector<float> *matrix){
    return;
}