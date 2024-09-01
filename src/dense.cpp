#include <cstdio>
#include "layers.h"
#include "initialization.h"
#include "activation.h"


Dense::Dense(int output_size, ActivationFn *activation){
    this->type = LayerType::DENSE;
    this->info.layer_name = LAYER_NAMES[this->type];
    this->activation = (activation == nullptr) ? new None() : activation;
    this->info.activation_fn_name = this->activation->get_name();
    this->info.output_neurons = output_size;
    

    this->is_initialized = false;
}

void Dense::initialize(int input_size){
    this->info.input_neurons = input_size;
    this->info.weights_count = input_size * this->info.output_neurons;
    this->info.biases_count = this->info.output_neurons;
    this->info.param_count = this->info.weights_count + this->info.biases_count;

    this->info.weight_bytes_count = this->info.param_count*sizeof(float); 
    this->info.bias_bytes_count = this->info.biases_count*sizeof(float);

    this->weights = std::vector<std::vector<float> >(
        this->info.output_neurons,
        std::vector<float>(this->info.input_neurons, 0)
    );
    glorot_uniform(&this->weights, this->info.input_neurons, this->info.output_neurons);

    // Creates a matrix of size (output_neurons x input_neurons)
    this->weight_gradients = std::vector<std::vector<float> >(
        this->info.output_neurons, 
        std::vector<float>(this->info.input_neurons, 0)
    );

    this->biases = std::vector<float>(this->info.output_neurons, 0);
    
    this->output = std::vector<std::vector<float> >(1, std::vector<float>(this->info.output_neurons, 0));

    this->is_initialized = true;
}

void Dense::forward(std::vector<std::vector<float> > *input){

    if(input->size() != this->output.size()){
        // Resize output to match input batch size
        this->output = std::vector<std::vector<float> >(input->size(), std::vector<float>(this->info.output_neurons, 0));
    }

    for(int batch = 0; batch < input->size(); batch++){
        for(int perceptron = 0; perceptron < this->info.output_neurons; perceptron++){
            for(int w_id = 0; w_id < this->weights[perceptron].size(); w_id++){
                this->output[batch][perceptron] += input->at(batch)[w_id] * this->weights[perceptron][w_id];
            }

            this->output[batch][perceptron] += this->biases[perceptron];
        }

        if(this->activation != nullptr){
            this->activation->forward(&this->output[batch]);
        }
    }
}

LayerInfo Dense::get_info(){
    return this->info;
}

void Dense::backward(std::vector<std::vector<float> > *grads){

    // TODO: Implement backpropagation

}