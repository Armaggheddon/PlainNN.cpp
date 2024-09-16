#ifndef MODEL_LAYERS_INPUT_H
#define MODEL_LAYERS_INPUT_H

#include "layer.h"

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

#endif // MODEL_LAYERS_INPUT_H