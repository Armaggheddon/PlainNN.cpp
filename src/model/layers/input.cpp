#include <cstdio>
#include <vector>
#include "layers.h"
#include <stdexcept>

Input::Input(int output_size){

    this->type = LayerType::INPUT;
    this->activation = new None();

    this->info = LayerInfo();
    this->info.layer_name = LAYER_NAMES[this->type];
    this->info.activation_fn_name = this->activation->get_name();
    this->info.input_neurons = output_size;
    this->info.output_neurons = output_size;
    this->info.param_count = 0;
    this->info.weights_count = 0;
    this->info.biases_count = 0;
    this->info.weight_bytes_count = 0;
    this->info.bias_bytes_count = 0;
}

void Input::initialize(int input_size){
    // No initialization needed for input layer, 
    // the output size is already set in the constructor
    // just check that input_size == output_size

    if(input_size != this->info.input_neurons){
        std::printf("Input size does not match output size\n");
        throw std::invalid_argument("Input size does not match output size");
    }
}

void Input::forward(std::vector<std::vector<float> > *input){
    if(input->size() != this->output.size()){
        this->output = std::vector<std::vector<float> >(input->size(), std::vector<float>(this->output.size(), 0));
    }

    for(int batch = 0; batch < input->size(); batch++){
        for(int i = 0; i < this->output.size(); i++){
            this->output[batch][i] = input->at(batch)[i];
        }
    }
}

void Input::backward(std::vector<std::vector<float> > *input){
    // No backward pass for input layer
    return;
}

LayerInfo Input::get_info(){
    return this->info;
}