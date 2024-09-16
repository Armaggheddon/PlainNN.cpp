#ifndef ACTIVATION_FNC_H
#define ACTIVATION_FNC_H

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


#endif // ACTIVATION_FNC_H