#include "lr_scheduler.hpp"

StepLR::StepLR(double gamma, int step_size){
    this->gamma = gamma;
    this->step_size = step_size;
}

void StepLR::step(double& learning_rate, int epoch){
    if(epoch % this->step_size == 0){
        learning_rate *= this->gamma;
    }
}

