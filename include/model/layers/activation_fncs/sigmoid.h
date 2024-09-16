#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation_fnc.h"

class Sigmoid : public ActivationFn{
    public:
        Sigmoid();

        double forward(const double input);

        double backward(const double input);
};

#endif // SIGMOID_H