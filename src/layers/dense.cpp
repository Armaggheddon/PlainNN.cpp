#include "layers.h"
#include "activation_fncs.h"
#include "initialization.h"

#include <vector>

Dense::Dense(int input_size, int output_size, ActivationFn* activation){

    this->input_size = input_size;
    this->output_size = output_size;

    this->layer_type = LayerType::DENSE;
    this->activation_fn = activation;

    this->output = Tensor({output_size});
    this->weights = Tensor({input_size, output_size}, true);
    this->d_weights = Tensor({input_size, output_size});
    this->biases = Tensor({output_size});
    this->d_biases = Tensor({output_size});

    this->is_initialized = true;
}

Dense::Dense(int output_size, ActivationFn* activation){

    this->output_size = output_size;

    this->layer_type = LayerType::DENSE;
    this->activation_fn = activation;

    this->output = Tensor({output_size});

    this->is_initialized = false;
}

void Dense::initialize(std::vector<int> input_shape){

    this->input_size = input_shape[0];

    this->weights = Tensor({input_size, output_size}, true);
    this->d_weights = Tensor({input_size, output_size});
    this->biases = Tensor({output_size});
    this->d_biases = Tensor({output_size});

    this->is_initialized = true;
}

Tensor* Dense::get_params(){
    return &this->weights;
}

std::vector<double> Dense::get_saveable_params(){
    std::vector<double> saveable_weights;
    
    double* weights_data = this->weights.data();
    for(int i=0; i<this->weights.size(); i++){
        saveable_weights.push_back(weights_data[i]);
    }

    double* biases_data = this->biases.data();
    for(int i=0; i<this->biases.size(); i++){
        saveable_weights.push_back(biases_data[i]);
    }

    return saveable_weights;
}

void Dense::load_params( std::vector<double>& params){
    int idx = 0;

    // todo assert initialized

    for(int i = 0; i < this->input_size; i++){
        for(int j = 0; j < this->output_size; j++){
            this->weights[i*this->output_size + j ] = params[idx++];
        }
    }

    for(int i = 0; i < this->output_size; i++){
        this->biases[i] = params[idx++];
    }
}

Tensor& Dense::forward(Tensor& input){

    // TODO: assert input shape is same as weights shape

    for(int j = 0; j< this->output_size; j++){
        this->output[j] = this->biases[j];
        for(int i=0; i<this->input_size; i++){
            this->output[j] += input[i] * this->weights[i*this->output_size + j];
        }
        this->output[j] = this->activation_fn->forward(this->output[j]);
    } 

    return output;
}

Tensor Dense::backward(
        Tensor* prev_output, 
        Tensor* next_weights,
        Tensor* next_grad){
    
    Tensor d_err = Tensor({this->output_size});
    Tensor grads = Tensor({this->output_size});

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){

        if(next_weights == nullptr){
            // If next_grad is null, it means that this is the last layer
            // and the next layer is the output layer. In this case, the
            // next_grad is the gradient of the loss function with respect
            // to the output of this layer.
            d_err[perceptron] = (*next_grad)[perceptron] - this->output[perceptron];
            grads[perceptron] += d_err[perceptron] * this->activation_fn->backward(this->output[perceptron]);
        } else{
            // If next_grad is not null, it means that this is not the last
            // layer and the next layer is not the output layer. In this case,
            // the next_grad is the gradient of the loss function with respect
            // to the output of the next layer.
            d_err[perceptron] = 0;
            int next_layer_size = next_grad->size();
            int offset = perceptron * next_layer_size;
            for(int next_perceptron = 0; next_perceptron < next_layer_size; next_perceptron++){
                d_err[perceptron] += (*next_grad)[next_perceptron] * (*next_weights)[offset + next_perceptron];
            }
            grads[perceptron] += d_err[perceptron] * this->activation_fn->backward(this->output[perceptron]);
        }
    }

    // Accumulate the gradients for the weights and biases
    for(int perceptron = 0; perceptron < this->input_size; perceptron++){
        int offset = perceptron * this->output_size;
        for(int weight = 0; weight < this->output_size; weight++){
            this->d_weights[offset + weight] += grads[weight] * (*prev_output)[perceptron];
        }
    }

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){
        this->d_biases[perceptron] += grads[perceptron];
    }

    return grads;
}

void Dense::step(double learning_rate, int batch_size){
    // Update the weights and biases
    for(int perceptron = 0; perceptron < this->input_size; perceptron++){
        int offset = perceptron * this->output_size;
        for(int weight = 0; weight < this->output_size; weight++){
            this->weights[offset + weight] += learning_rate * this->d_weights[offset + weight] / batch_size;
        }
    }

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){
        this->biases[perceptron] += learning_rate * this->d_biases[perceptron] / batch_size;
    }

    // Reset the gradients
    d_weights.clear();
    d_biases.clear();
}