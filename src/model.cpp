#include <iostream>
#include <cstdio>
#include <vector>
#include "model.h"

Model::Model(){
    this->layers = std::vector<Layer*>();
}

void Model::add(Layer *layer){
    LayerSummary layer_info = layer->get_summary();
    if(this->layers.size() == 0 && (layer_info.layer_name.compare("input"))){
        std::printf("Error: First layer must be Input layer\n");
        return;
    }
    if(this->layers.size() == 0) this->input_size = layer_info.output_size;
    this->layers.push_back(layer);
}

std::vector<std::vector<float> > Model::forward(std::vector<std::vector<float> > *input){
    // this->layers[0]->initialize(input->at(0).size());
    // this->layers[1]->initialize(this->layers[0]->output[0].size());
    this->layers[1]->forward(input);
    this->layers[2]->forward(&this->layers[1]->output);
    return this->layers[this->layers.size()-1]->output;
}

void Model::backward(){
    std::printf("Not implemented\n");
}

void Model::summary(){
    std::printf("%-10s%-10s%-10s\n", "Layer", "Output", "Param #");
    std::printf("------------------------------------------\n");
    float trainable_param_count = 0;
    for(int i=0; i<this->layers.size(); i++){
        LayerSummary summary = this->layers[i]->get_summary();
        std::string layer_name = summary.layer_name + "_" + std::to_string(i);
        std::string output_size = "(" + std::to_string(summary.batch_size) + ", " + std::to_string(summary.output_size) + ")";
        std::printf("%-10s%-10s%d\n", layer_name.c_str(), output_size.c_str(), summary.param_count);
        trainable_param_count += summary.param_count;
    }
    std::printf("------------------------------------------\n");
    std::printf("Total trainable parameters: %d\n", (int)trainable_param_count);
}

void Model::compile(){
    this->layers[1]->initialize(this->input_size);
    for(int i=2; i<this->layers.size(); i++){
        this->layers[i]->initialize(this->layers[i-1]->output[0].size());
    }
}

Layer* Model::operator[](int index){
    return this->layers[index];
}

