#include "layers.h"
#include "activation_fncs.h"

#include "initialization.h"

#include <cmath>
#include <vector>


Dense::Dense(int input_size, int output_size, ActivationFn* activation){

    this->layer_type = LayerType::DENSE;

    this->config.type = layer_type;
    this->config.input_size = input_size;
    this->config.output_size = output_size;
    this->config.activation = activation->name();
    this->params.activation_fn = activation;
    this->params.weights = std::vector<std::vector<double> >(
        input_size, std::vector<double>(output_size, 0));
    this->params.d_weights = std::vector<std::vector<double> >(
        input_size, std::vector<double>(output_size, 0));
    this->params.biases = std::vector<double>(output_size, 0);
    this->params.d_biases = std::vector<double>(output_size, 0);
    this->output = std::vector<double>(output_size, 0);

    GolorotInitialization::initialize(this->params.weights, input_size, output_size);

    this->is_initialized = true;
}

Dense::Dense(int output_size, ActivationFn* activation){

    this->layer_type = LayerType::DENSE;

    this->config.type = layer_type;

    this->config.output_size = output_size;
    this->config.activation = activation->name();
    this->params.activation_fn = activation;
    this->output = std::vector<double>(output_size, 0);

    this->is_initialized = false;
}

void Dense::initialize(int input_size){

    this->config.input_size = input_size;
    this->params.weights = std::vector<std::vector<double> >(
            input_size, std::vector<double>(this->config.output_size, 0));
    this->params.d_weights = std::vector<std::vector<double> >(
            input_size, std::vector<double>(this->config.output_size, 0));
    this->params.biases = std::vector<double>(this->config.output_size, 0);
    this->params.d_biases = std::vector<double>(this->config.output_size, 0);
    
    // Initialize the weights and biases
    GolorotInitialization::initialize(this->params.weights, input_size, this->config.output_size);

    this->is_initialized = true;
}

std::vector<double> Dense::get_saveable_params(){
    std::vector<double> saveable_weights;
    for(int i = 0; i < this->config.input_size; i++){
        for(int j = 0; j < this->config.output_size; j++){
            saveable_weights.push_back(this->params.weights[i][j]);
        }
    }
    for(int j = 0; j < this->config.output_size; j++){
        saveable_weights.push_back(this->params.biases[j]);
    }
    return saveable_weights;
}

void Dense::load_weights_and_biases_from_vector(std::vector<double>& params){
    int idx = 0;
    for(int i = 0; i < this->config.input_size; i++){
        for(int j = 0; j < this->config.output_size; j++){
            this->params.weights[i][j] = params[idx];
            idx++;
        }
    }
    for(int j = 0; j < this->config.output_size; j++){
        this->params.biases[j] = params[idx];
        idx++;
    }
}

std::vector<double> Dense::forward(std::vector<double>* input){
    for(int j = 0; j < this->config.output_size; j++){
        output[j] = this->params.biases[j];
        for(int i = 0; i < this->config.input_size; i++){
            output[j] += (*input)[i] * this->params.weights[i][j];
        }
        output[j] = this->params.activation_fn->forward(output[j]);
    }
    return output;
}

std::vector<std::vector<double> > Dense::backward(
                const std::vector<double>* prev_output, 
                const std::vector<std::vector<double> > *next_weights,
                const std::vector<double> *next_grad){
    
    std::vector<double> d_err(this->config.output_size, 0);
    std::vector<double> grads(this->config.output_size, 0);

    for(int perceptron = 0; perceptron < this->config.output_size; perceptron++){

        if(next_weights == nullptr){
            // If next_grad is null, it means that this is the last layer
            // and the next layer is the output layer. In this case, the
            // next_grad is the gradient of the loss function with respect
            // to the output of this layer.
            d_err[perceptron] = (*next_grad)[perceptron] - this->output[perceptron];
            grads[perceptron] += d_err[perceptron] * this->params.activation_fn->backward(this->output[perceptron]);
        } else{
            // If next_grad is not null, it means that this is not the last
            // layer and the next layer is not the output layer. In this case,
            // the next_grad is the gradient of the loss function with respect
            // to the output of the next layer.
            d_err[perceptron] = 0;
            for(int next_perceptron = 0; next_perceptron < next_grad->size(); next_perceptron++){
                d_err[perceptron] += (*next_grad)[next_perceptron] * (*next_weights)[perceptron][next_perceptron];
            }
            grads[perceptron] += d_err[perceptron] * this->params.activation_fn->backward(this->output[perceptron]);
        }
    }

    // Accumulate the gradients for the weights and biases
    for(int perceptron = 0; perceptron < this->config.input_size; perceptron++){
        for(int weight = 0; weight < this->config.output_size; weight++){
            this->params.d_weights[perceptron][weight] += grads[weight] * (*prev_output)[perceptron];
        }
    }

    for(int perceptron = 0; perceptron < this->config.output_size; perceptron++){
        this->params.d_biases[perceptron] += grads[perceptron];
    }

    return std::vector<std::vector<double> >(1, grads);
}

void Dense::step(double learning_rate, int batch_size){
    // Update the weights and biases
    for(int perceptron = 0; perceptron < this->config.input_size; perceptron++){
        for(int weight = 0; weight < this->config.output_size; weight++){
            this->params.weights[perceptron][weight] += learning_rate * this->params.d_weights[perceptron][weight] / batch_size;
        }
    }

    for(int perceptron = 0; perceptron < this->config.output_size; perceptron++){
        this->params.biases[perceptron] += learning_rate * this->params.d_biases[perceptron] / batch_size;
    }

    // Reset the gradients
    std::fill(this->params.d_weights.begin(), this->params.d_weights.end(), std::vector<double>(this->config.input_size, 0));
    std::fill(this->params.d_biases.begin(), this->params.d_biases.end(), 0);
}