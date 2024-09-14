#include "layers.h"

#include <vector>
#include <stdexcept>

#include "initialization.h"

Input::Input(LayerConfig config){

    this->layer_type = LayerType::INPUT;

    this->config = config;
    this->output = std::vector<double>(config.output_size, 0);
}

Input::Input(int input_size){

    this->layer_type = LayerType::INPUT;

    this->config.type = LayerType::INPUT;

    this->config.input_size = input_size;
    this->config.output_size = input_size;
    this->output = std::vector<double>(input_size, 0);

    this->is_initialized = true;
}

std::vector<double> Input::forward(std::vector<double>* input){
    this->output = *input;
    return this->output;
}

std::vector<std::vector<double> > Input::backward(
                const std::vector<double>* prev_output, 
                const std::vector<std::vector<double> > *next_weights,
                const std::vector<double> *next_grad){
    
    // The input layer does not have any weights or biases, so there is no
    // need to calculate the gradients. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

void Input::step(double learning_rate, int batch_size){
    // The input layer does not have any weights or biases, so there is no
    // need to update them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}