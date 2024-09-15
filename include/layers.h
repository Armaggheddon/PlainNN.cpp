#ifndef LAYERS_H
#define LAYERS_H

#include <vector>

#include "tensor.h"
#include "activation_fncs.h"


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

class Input : public Layer{
    public:
        Input(std::initializer_list<int> shape);

        Tensor& forward( Tensor& input);
        Tensor backward( Tensor* prev_output,  Tensor* next_weights,  Tensor* next_grad);
        void step(double learning_rate, int batch_size);
        std::vector<double> get_saveable_params();
        void load_params( std::vector<double>& params);

        void initialize(std::vector<int> input_shape);
};

class Dense : public Layer{
    public:
        Dense(int output_size, ActivationFn* activation_fn);
        Dense(int input_size, int output_size, ActivationFn* activation_fn);

        void initialize(std::vector<int> input_shape);
        Tensor* get_params() override;
        Tensor& forward(Tensor& input);
        Tensor backward(Tensor* prev_output, Tensor* next_weights, Tensor* next_grad);
        void step(double learning_rate, int batch_size);
        std::vector<double> get_saveable_params();
        void load_params( std::vector<double>& params);

    private:
        int input_size, output_size; 
        Tensor weights;
        Tensor d_weights;
        Tensor biases;
        Tensor d_biases;
        ActivationFn* activation_fn;
};

#endif // LAYERS_H