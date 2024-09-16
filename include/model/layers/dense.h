#ifndef MODEL_LAYERS_DENSE_H
#define MODEL_LAYERS_DENSE_H

#include "layer.h"

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

#endif // MODEL_LAYERS_DENSE_H