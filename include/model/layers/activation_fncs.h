#ifndef MODEL_LAYERS_ACTIVATION_FNC_H
#define MODEL_LAYERS_ACTIVATION_FNC_H

#include <cmath>
#include <string>

enum ActivationType{
    NONE,
    RELU,
    SIGMOID,
    SOFTMAX
};

const std::string ACTIVATION_NAMES[] = {
    "None",
    "ReLU",
    "Sigmoid",
    "Softmax"
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
ActivationFn* get_activation_fn_from_name(std::string name);


class ReLU : public ActivationFn{
    public:
        ReLU();

        double forward(const double input);

        double backward(const double input);
};


class Sigmoid : public ActivationFn{
    public:
        Sigmoid();

        double forward(const double input);

        double backward(const double input);
};

class None : public ActivationFn{
    public:
        None();

        double forward(const double input);

        double backward(const double input);
};

#endif // MODEL_LAYERS_ACTIVATION_FNC_H