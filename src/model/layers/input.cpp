#include <vector>
#include <stdexcept>
#include <initializer_list>

#include "input.h"
#include "tensor.h"

Input::Input(std::initializer_list<int> shape){


    this->layer_type = LayerType::INPUT;

    this->output = Tensor(shape);

    this->is_initialized = true;
}

Tensor& Input::forward( Tensor& input){
    this->output = input;
    return this->output;
}

Tensor Input::backward( Tensor* prev_output,  Tensor* next_weights,  Tensor* next_grad){
    
    // The input layer does not have any weights or biases, so there is no
    // need to calculate the gradients. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

void Input::step(double learning_rate, int batch_size){
    // The input layer does not have any weights or biases, so there is no
    // need to update them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}



std::vector<double> Input::get_saveable_params(){
    // The input layer does not have any weights or biases, so there is no
    // need to save them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

void Input::load_params( std::vector<double>& params){
    // The input layer does not have any weights or biases, so there is no
    // need to load them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

void Input::initialize(std::vector<int> input_shape){
    // The input layer does not have any weights or biases, so there is no
    // need to initialize them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}