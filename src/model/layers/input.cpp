#include <vector>
#include <stdexcept>
#include <initializer_list>

#include "layers.h"
#include "tensor.h"

Input::Input(std::initializer_list<int> shape, bool frozen){

    this->layer_type = LayerType::INPUT;

    this->output = Tensor(shape);

    this->is_frozen = frozen;
    this->is_initialized = true;
}

Input::Input(std::vector<int> shape, bool frozen){
    
    this->layer_type = LayerType::INPUT;

    this->output = Tensor(shape);

    this->is_frozen = frozen;
    this->is_initialized = true;
}

Tensor& Input::forward( Tensor& input){
    this->output = input;
    return this->output;
}

Tensor Input::backward(__attribute_maybe_unused__ Tensor* prev_output, __attribute_maybe_unused__ Tensor* next_weights, __attribute_maybe_unused__ Tensor* next_grad){
    
    // The input layer does not have any weights or biases, so there is no
    // need to calculate the gradients. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

void Input::step(__attribute_maybe_unused__ double learning_rate, __attribute_maybe_unused__ int batch_size){
    // The input layer does not have any weights or biases, so there is no
    // need to update them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}



std::vector<double> Input::get_saveable_params(){
    // The input layer does not have any weights or biases, so there is no
    // need to save them. Throw runtime exception with message

    return std::vector<double>();
}

void Input::load_params( __attribute_maybe_unused__ std::vector<double>& params){
    // The input layer does not have any weights or biases, so there is no
    // need to load them. Throw runtime exception with message

    return;
}

void Input::initialize( __attribute_maybe_unused__ std::vector<int> input_shape){
    // The input layer does not have any weights or biases, so there is no
    // need to initialize them. Throw runtime exception with message

    throw std::runtime_error("Input layer does not have weights or biases, so it does not have gradients.");
}

LayerSummary Input::get_summary(){
    LayerSummary summary;
    summary.layer_type = this->layer_type;
    summary.layer_name = LAYER_TYPE_NAMES[this->layer_type];
    summary.activation_fn = ACTIVATION_NAMES[ActivationType::NONE];
    summary.param_count = 0;
    summary.param_size = 0;
    summary.layer_shape = this->output.shape();

    return summary;
}