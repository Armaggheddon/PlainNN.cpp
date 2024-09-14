#ifndef LAYERS_H
#define LAYERS_H

#include "activation_fncs.h"

#include <vector>
#include <string>

enum LayerType{
    INPUT,
    DENSE
};

const std::string LAYER_TYPE_NAMES[] = {
    "Input",
    "Dense"
};

typedef struct{
    LayerType type;
    std::string name;
    std::string name_custom;
    std::string activation;
    int input_size;
    int output_size;
}  LayerConfig;

typedef struct {
    std::vector<std::vector<double> > weights;
    std::vector<std::vector<double> > d_weights;
    std::vector<double> biases;
    std::vector<double> d_biases;
    ActivationFn* activation_fn;
} LayerParams;


class Layer{
    public:

        bool is_initialized = false;
        std::vector<double> output;
        LayerConfig config;
        LayerParams params;

        ~Layer(){};

        virtual std::vector<double> forward(std::vector<double>* input) = 0;
        virtual std::vector<std::vector<double> > backward(
                        const std::vector<double> *prev_output = nullptr, 
                        const std::vector<std::vector<double> > *next_weights = nullptr,
                        const std::vector<double> *next_grad = nullptr) = 0;
        
        virtual void step(double learning_rate, int batch_size) = 0;
        virtual std::vector<double> get_saveable_params() = 0;
        virtual void load_weights_and_biases_from_vector(std::vector<double>& params) = 0;

        std::string name(){return LAYER_TYPE_NAMES[layer_type];};

    protected:
        LayerType layer_type;
};

class Dense : public Layer{
    public:
        Dense(int input_size, int output_size, ActivationFn* activation);
        Dense(int output_size, ActivationFn* activation);
        std::vector<double> forward(std::vector<double>* input);
        std::vector<std::vector<double> > backward(
                        const std::vector<double>* prev_output, 
                        const std::vector<std::vector<double> > *next_weights,
                        const std::vector<double> *next_grad);
        void step(double learning_rate, int batch_size);

        void initialize(int input_size);
        std::vector<double> get_saveable_params();
        void load_weights_and_biases_from_vector(std::vector<double>& params);
};

class Input : public Layer{
    public:
        Input(LayerConfig config);
        Input(int input_size);
        std::vector<double> forward(std::vector<double>* input);
        
        // This function is not used in the Input layer, 
        // but it is necessary to implement it. It simply 
        // returns an empty vector.
        std::vector<std::vector<double> > backward(
                        const std::vector<double>* prev_output, 
                        const std::vector<std::vector<double> > *next_weights,
                        const std::vector<double> *next_grad);

        void step(double learning_rate, int batch_size);
        std::vector<double> get_saveable_params(){return std::vector<double>();};
        void load_weights_and_biases_from_vector(std::vector<double>& params){};
};

#endif // LAYERS_H