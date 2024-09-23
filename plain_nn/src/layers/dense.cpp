#include "layers.hpp"
#include "activation_fncs.hpp"

#include <stdexcept>
#include <vector>

Dense::Dense(int input_size, int output_size, ActivationFn* activation, bool frozen){

    this->input_size = input_size;
    this->output_size = output_size;

    this->layer_type = LayerType::DENSE;
    this->activation_fn = activation;

    this->output = Tensor({output_size});
    this->weights = Tensor({input_size, output_size}, true);
    this->d_weights = Tensor({input_size, output_size});
    this->biases = Tensor({output_size});
    this->d_biases = Tensor({output_size});

    this->is_frozen = frozen;
    this->is_initialized = true;
}

Dense::Dense(int output_size, ActivationFn* activation, bool frozen){

    this->output_size = output_size;

    this->layer_type = LayerType::DENSE;
    this->activation_fn = activation;

    this->output = Tensor({output_size});

    this->is_frozen = frozen;
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

    size_t params_count = this->input_size * this->output_size + this->output_size;
    if(params.size() != params_count){
        throw std::runtime_error("Invalid number of parameters, expected " + std::to_string(params_count) + " got " + std::to_string(params.size()));
    }

    for(int i = 0; i < this->input_size; i++){
        for(int j = 0; j < this->output_size; j++){
            this->weights[i*this->output_size + j ] = params[idx++];
        }
    }

    for(int i = 0; i < this->output_size; i++){
        this->biases[i] = params[idx++];
    }

    this->is_initialized = true;
}

Tensor& Dense::forward(Tensor& input){

    double *_input = input.data();
    double *_output = this->output.data();
    double *_weights = this->weights.data();
    double *_biases = this->biases.data();

    for(int j = 0; j< this->output_size; j++){
        _output[j] = _biases[j];
        for(int i=0; i<this->input_size; i++){
            _output[j] += _input[i] * _weights[i*this->output_size + j];
        }
        // _output[j] = this->activation_fn->forward(_output[j]);
    } 
    output = this->activation_fn->forward(output);

    return output;
}

Tensor Dense::backward(
        Tensor* prev_output, 
        Tensor* next_weights,
        Tensor* next_grad){
    
    Tensor d_err = Tensor({this->output_size});
    Tensor grads = Tensor({this->output_size});

    // Taking a local reference directly to the data
    // significantly improves the performance
    double* _prev_output = (prev_output == nullptr) ? nullptr : prev_output->data();
    double* _next_weights = (next_weights == nullptr) ? nullptr : next_weights->data();
    double* _next_grad = (next_grad == nullptr) ? nullptr : next_grad->data();
    
    double* _d_err = d_err.data();
    double* _grads = grads.data();
    double* _output = this->output.data();
    double* _d_weights = this->d_weights.data();
    double* _d_biases = this->d_biases.data();

    Tensor act_fn_der = this->activation_fn->backward(this->output);
    double* _act_fn_der = act_fn_der.data();

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){

        if(next_weights == nullptr){
            // If next_grad is null, it means that this is the last layer
            // and the next layer is the output layer. In this case, the
            // next_grad is the gradient of the loss function with respect
            // to the output of this layer.
            _d_err[perceptron] = _next_grad[perceptron] - _output[perceptron];
            _grads[perceptron] += _d_err[perceptron] * _act_fn_der[perceptron];
        } else{
            // If next_grad is not null, it means that this is not the last
            // layer and the next layer is not the output layer. In this case,
            // the next_grad is the gradient of the loss function with respect
            // to the output of the next layer.
            _d_err[perceptron] = 0;
            int next_layer_size = next_grad->size();
            int offset = perceptron * next_layer_size;
            for(int next_perceptron = 0; next_perceptron < next_layer_size; next_perceptron++){
                _d_err[perceptron] += _next_grad[next_perceptron] * _next_weights[offset + next_perceptron];
            }
            _grads[perceptron] += _d_err[perceptron] * _act_fn_der[perceptron];
        }
    }

    // Accumulate the gradients for the weights and biases
    for(int perceptron = 0; perceptron < this->input_size; perceptron++){
        int offset = perceptron * this->output_size;
        for(int weight = 0; weight < this->output_size; weight++){
            _d_weights[offset + weight] += _grads[weight] * _prev_output[perceptron];
        }
    }

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){
        _d_biases[perceptron] += _grads[perceptron];
    }

    return grads;
}

void Dense::step(double learning_rate, int batch_size){
    
    double* _d_weights = this->d_weights.data();
    double* _weights = this->weights.data();
    double* _d_biases = this->d_biases.data();
    double* _biases = this->biases.data();

    for(int perceptron = 0; perceptron < this->input_size; perceptron++){
        int offset = perceptron * this->output_size;
        for(int weight = 0; weight < this->output_size; weight++){
            _weights[offset + weight] += learning_rate * _d_weights[offset + weight] / batch_size;
        }
    }

    for(int perceptron = 0; perceptron < this->output_size; perceptron++){
        _biases[perceptron] += learning_rate * _d_biases[perceptron] / batch_size;
    }

    // Reset the gradients
    d_weights.clear();
    d_biases.clear();
}

LayerSummary Dense::get_summary(){
    LayerSummary summary;
    summary.layer_type = this->layer_type;
    summary.layer_name = LAYER_TYPE_NAMES[this->layer_type];
    summary.activation_fn = this->activation_fn->name();

    summary.param_count = this->input_size * this->output_size + this->output_size;
    summary.param_size = sizeof(double);

    summary.layer_shape = weights.shape();
    return summary;
}

