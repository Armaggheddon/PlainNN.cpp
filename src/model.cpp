#include <iostream>
#include <cstdio>
#include <vector>
#include <cmath>
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

std::vector<std::vector<double> > Model::forward(std::vector<std::vector<double> > *input){
    this->layers[1]->forward(input);
    this->layers[2]->forward(&this->layers[1]->output);
    return this->layers[2]->output;
}

void Model::train(DataLoader *x, int epochs, int batch_size, double learning_rate){

    for(int i=0; i<epochs; i++){
        std::printf("Epoch %d\n", i);
        std::vector<Data> batch_data = x->get_batch(batch_size);

        std::vector<std::vector<double> > batch_inputs(batch_size, std::vector<double>(batch_data.size(), 0));
        for(int j=0; j<batch_data.size(); j++){
            batch_inputs[j] = batch_data[j].input;
        }

        std::vector<std::vector<double> > result = this->forward(&batch_inputs);

        // calculate the loss for this step using mse

        // loss = 1/k * SUM[j=0 -> k](aj - pj)^2 where aj is the actual value and pj is the predicted value
        // loss for batch = 1/N * SUM[i=0 -> N](loss) where N is the number of samples in the batch

        // loss vector representing the loss per output node averaged over the batches
        std::vector<double> loss(result[0].size(), 0);

        for(int batch = 0; batch < result.size(); batch++){
            int target_id = batch_data[batch].label;
            std::vector<double> target_vec = this->get_one_hot(target_id, result[batch].size());
            for(int out_idx = 0; out_idx < result[batch].size(); out_idx++){
                loss[out_idx] += 1/static_cast<double>(result[batch].size())*std::pow(target_vec[out_idx] - result[batch][out_idx], 2);
            }
        }

        double total_loss = 0;
        for(int j = 0; j< loss.size(); j++){
            loss[j] = loss[j]/static_cast<double>(batch_size);
            total_loss += loss[j];
            std::printf("\tLoss[%d]: %f\n", j, loss[j]);
        }
        std::printf("\tTotal Loss: %f\n", total_loss);

        // backpropagation
        // calculate the gradients for the last layer


        
    }
}

std::vector<double> Model::get_one_hot(int label, int size){
    std::vector<double> one_hot(size, 0);
    one_hot[label] = 1;
    return one_hot;
}

void Model::save(std::string filename){
    // TODO: Implement model saving
}

void Model::load(std::string filename){
    // TODO: Implement model loading
}

void Model::summary(){
    std::printf("%-10s%-10s%-10s\n", "Layer", "Output", "Param #");
    std::printf("------------------------------------------\n");
    double trainable_param_count = 0;
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

