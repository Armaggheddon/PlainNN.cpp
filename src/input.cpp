#include <cstdio>
#include <vector>
#include "layers.h"

Input::Input(int output_size){
    this->summary = LayerSummary();
    this->summary.layer_name = "input";
    this->summary.output_size = output_size;
    this->summary.batch_size = 1;
    this->summary.param_count = 0;
    this->summary.param_size = 0;
    this->weights = std::vector<std::vector<float> >(output_size, std::vector<float>(output_size, 0));
}

void Input::initialize(int input_size){
    
}

void Input::forward(std::vector<std::vector<float> > *input){
    if(input->size() != this->summary.batch_size){
        this->summary.batch_size = input->size();
        this->output = std::vector<std::vector<float> >(this->summary.batch_size, std::vector<float>(this->summary.output_size, 0));
    }

    for(int batch = 0; batch < this->summary.batch_size; batch++){
        for(int i = 0; i < this->summary.input_size; i++){
            this->output[batch][i] = input->at(batch)[i];
        }
    }
}

void Input::backward(std::vector<std::vector<float> > *input){
    std::printf("Not implemented\n");
}

LayerSummary Input::get_summary(){
    return this->summary;
}