#include <cstdio>
#include "layers.h"
#include "initialization.h"


Dense::Dense(int output_size, act_func activation){
    this->summary.layer_name = "dense";
    this->summary.output_size = output_size;
    this->activation = activation;
}

void Dense::initialize(int input_size){
    this->summary.input_size = input_size;
    this->summary.batch_size = 1;
    this->summary.param_count = input_size*summary.output_size + summary.output_size;
    this->summary.param_size = this->summary.param_count*sizeof(float); 

    this->weights = std::vector<std::vector<float> >(
        summary.output_size, 
        std::vector<float>(input_size, 0)
    );
    glorot_uniform(&this->weights, input_size, this->summary.output_size);
    
    this->grad = std::vector<std::vector<float> >(
        summary.output_size, 
        std::vector<float>(input_size, 0)
    );
    glorot_uniform(&this->grad, 0, summary.output_size);

    this->bias = std::vector<float>(summary.output_size, 0);
    
    this->output = std::vector<std::vector<float> >(1, std::vector<float>(summary.output_size, 0));

    this->initialized = true;
}

void Dense::forward(std::vector<std::vector<float> > *input){

    // Check if input size matches output size, if not, 
    // output size has to be changed to match the current run
    if(input->size() != this->summary.batch_size){
        this->summary.batch_size = input->size();
        this->output = std::vector<std::vector<float> >(this->summary.batch_size, std::vector<float>(this->summary.output_size, 0));
    }

    for(int batch = 0; batch < this->summary.batch_size; batch++){
        for(int perceptron = 0; perceptron < this->weights.size(); perceptron++){
            for(int w_id = 0; w_id < this->summary.input_size; w_id++){
                this->output[batch][perceptron] += input->at(batch)[w_id] * this->weights[perceptron][w_id];
            }

            this->output[batch][perceptron] += this->bias[perceptron];
        }

        if(this->activation != NULL){
            this->activation(&this->output[batch]);
        }
    }
}

LayerSummary Dense::get_summary(){
    return this->summary;
}

void Dense::backward(std::vector<std::vector<float> > *grads){

    for(int batch = 0; batch < this->summary.batch_size; batch++){
        for(int perceptron = 0; perceptron < this->summary.output_size; perceptron++){
            // std::printf("Perceptron: %d\n", perceptron);
            for(int w_id = 0; w_id < this->summary.input_size; w_id++){
                this->grad[batch][w_id] += grads->at(batch)[perceptron] * this->weights[perceptron][w_id];
                this->weights[perceptron][w_id] += grads->at(batch)[perceptron] * this->output[batch][perceptron];
            }

            this->bias[perceptron] += grads->at(batch)[perceptron];
        }
    }

}