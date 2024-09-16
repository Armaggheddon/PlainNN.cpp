#ifndef LAYERS_H
#define LAYERS_H

#include <vector>

#include "tensor.h"
#include "activation_fnc.h"


enum LayerType{
    INPUT,
    DENSE
};

const std::string LAYER_TYPE_NAMES[] = {
    "Input",
    "Dense"
};


class Layer{
    public:

        bool is_initialized = false;
        Tensor output;
        LayerType layer_type;

        ~Layer(){};

        virtual Tensor& forward(Tensor& input) = 0;
        virtual Tensor backward(Tensor*  prev_output, Tensor* next_weights, Tensor* next_grad) = 0;
        virtual void step(double learning_rate, int batch_size) = 0;
        virtual std::vector<double> get_saveable_params() = 0;
        virtual void load_params(std::vector<double>& params) = 0;

        virtual Tensor* get_params(){return new Tensor();};

        virtual void initialize(std::vector<int> input_shape) = 0;

        std::string name(){return LAYER_TYPE_NAMES[layer_type];};
};

#endif // LAYERS_H