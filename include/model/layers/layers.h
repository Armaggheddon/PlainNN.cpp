#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <string>
#include "activation.h"

enum LayerType{
    INPUT = 0,
    DENSE = 1
};

const std::string LAYER_NAMES[] = {
    "input",
    "dense"
};

typedef struct{

    std::string layer_name;
    std::string activation_fn_name;
    int input_neurons, output_neurons;
    long int param_count;
    int weights_count, biases_count;
    long int weight_bytes_count, bias_bytes_count; // in bytes

} LayerInfo;


typedef struct{
    std::string layer_name;
    std::string activation_fn_name;
    int input_size, batch_size, output_size;
    int param_count;
    long int param_size; // in bytes
} LayerSummary;

class Layer{
    public:
        std::vector<std::vector<float> > output;
        std::vector<std::vector<float> > weights;
        std::vector<std::vector<float> > weight_gradients;
        std::vector<float> biases;
        ActivationFn *activation;
        
        virtual ~Layer(){};
        virtual void initialize(int input_size) = 0;
        virtual void forward(std::vector<std::vector<float> > *input) = 0;
        virtual void backward(std::vector<std::vector<float> > *input) = 0;
        virtual ActivationFn* get_activation() { return this->activation; }
        virtual void set_activation(ActivationFn *activation) { this->activation = activation; }
        virtual std::string get_name(){ return LAYER_NAMES[this->type]; }
        virtual LayerInfo get_info() = 0;
    protected:
        LayerType type;
        bool is_initialized = false;
        LayerInfo info;
};

class Dense: public Layer{
    public:
        Dense(int output_size, ActivationFn *activation = nullptr);
        virtual void initialize(int input_size);
        virtual void forward(std::vector<std::vector<float> > *input);
        virtual void backward(std::vector<std::vector<float> > *input);
        virtual LayerInfo get_info();
};

class Input: public Layer{
    public: 
        Input(int output_size);
        virtual void initialize(int input_size);

        /**
         * @brief Forward pass for the input layer, 
         * since the input layer performs no operations on the
         * data, this call simply copies the input data to the
         * output vector.
         * 
         * @param input: the batch of inputs
         */
        virtual void forward(std::vector<std::vector<float> > *input);
        virtual void backward(std::vector<std::vector<float> > *input);
        virtual LayerInfo get_info();
};

#endif