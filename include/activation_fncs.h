#ifndef ACTIVATION_FNCS_H
#define ACTIVATION_FNCS_H

#include <cmath>
#include <string>


enum ActivationType{
    NONE,
    RELU,
    SIGMOID,
};

const std::string ACTIVATION_NAMES[] = {
    "None",
    "ReLU",
    "Sigmoid"
};

class ActivationFn{
    public:
        ~ActivationFn(){};

        ActivationType fn_type;

        virtual double forward(const double input) = 0;
        virtual double backward(const double input) = 0;

        std::string name(){return ACTIVATION_NAMES[fn_type];}
        ActivationType type(){return fn_type;}
};

class ReLU : public ActivationFn{
    public:
        ReLU(){
            this->fn_type = ActivationType::RELU;
        }
        double forward(const double input){
            return std::max(0.0, input);
        }
        double backward(const double input){
            return input > 0 ? 1 : 0;
        }
};

class Sigmoid : public ActivationFn{
    public:
        Sigmoid(){
            this->fn_type = ActivationType::SIGMOID;
        }
        double forward(const double input){
            return 1 / (1 + std::exp(-input));
        }
        double backward(const double input){
            return input * (1 - input);
        }
};

#endif // ACTIVATION_FNCS_H