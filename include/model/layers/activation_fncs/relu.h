#ifndef RELU_H
#define RELU_H

#include "activation_fnc.h"

class ReLU : public ActivationFn{
    public:
        ReLU();

        double forward(const double input);

        double backward(const double input);
};

#endif // RELU_H