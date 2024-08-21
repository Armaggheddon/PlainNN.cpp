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

std::vector<std::vector<float> > Model::forward(std::vector<std::vector<float> > *input){
    this->layers[1]->forward(input);
    this->layers[2]->forward(&this->layers[1]->output);
    return this->layers[2]->output;
}

void Model::train(DataLoader *x, int epochs, int batch_size, float learning_rate){
    Data test_sample = x->get_sample();
    std::vector<std::vector<float> > test_input(1, std::vector<float>(784, 0));
    for(int i=0; i<test_sample.input.size(); i++){
        test_input[0][i] = test_sample.input[i];
    }
    for(int i=0; i<epochs; i++){
        std::printf("Epoch %d\n", i);
        std::vector<Data> batch_data = x->get_batch(batch_size);

        std::vector<std::vector<float> > batch_inputs(batch_size, std::vector<float>(batch_data.size(), 0));
        for(int j=0; j<batch_data.size(); j++){
            batch_inputs[j] = batch_data[j].input;
        }

        std::vector<std::vector<float> > result = this->forward(&batch_inputs);

        std::vector<std::vector<float> > target(batch_size, std::vector<float>(10, 0));
        for(int j=0; j<batch_data.size(); j++){
            target[j] = this->get_one_hot(batch_data[j].label, 10);
        }
        std::vector<float> mse = this->mse(result, target);
        std::vector<float> loss = this->loss(result, batch_inputs);

        float avg_mse = 0;
        for(int j=0; j<mse.size(); j++){
            avg_mse += mse[j];
        }
        std::printf("Average MSE: %f\n", avg_mse);

        std::vector<std::vector<float> > deltas(this->layers[2]->weights.size(), std::vector<float>(layers[2]->weights[0].size(), 0));
        // calculate delta for last layer
        for(int batch = 0; batch<result.size(); batch++){
            for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
                for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                    deltas[perceptron][weight] += (target[batch][perceptron] - result[batch][perceptron]) * result[batch][perceptron] * (1 - result[batch][perceptron]) * this->layers[1]->output[batch][weight];
                }
            }
        }

        // average deltas for batch size
        for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                deltas[perceptron][weight] /= static_cast<float>(result.size());
            }
        }

        // update weights for last layer
        for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                this->layers[2]->weights[perceptron][weight] += learning_rate * deltas[perceptron][weight];
            }
        }

        // calculate delta for second last layer
        std::vector<std::vector<float> > deltas_2(this->layers[1]->weights.size(), std::vector<float>(layers[1]->weights[0].size(), 0));
        for(int batch = 0; batch<result.size(); batch++){
            for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
                for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                    float eo1_net01 = 0;
                    for(int i=0; i<this->layers[2]->weights.size(); i++){
                        eo1_net01 += (target[batch][i] - result[batch][i]) * this->layers[2]->output[batch][i] * (1 - this->layers[2]->output[batch][i]);
                    }
                    deltas_2[perceptron][weight] += eo1_net01 * 1 * this->layers[1]->weights[perceptron][weight] * batch_inputs[batch][weight];
                }
            }
        }

        std::printf("HERE\n");

        // average deltas for batch size
        for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                deltas_2[perceptron][weight] /= static_cast<float>(result.size());
            }
        }

        // update weights for second last layer
        for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                this->layers[1]->weights[perceptron][weight] -= learning_rate * deltas_2[perceptron][weight];
            }
        }

        // run forward pass
        result = this->forward(&test_input);
        target.clear();
        target.push_back(this->get_one_hot(test_sample.label, 10));
        mse = this->mse(result, target);

        avg_mse = 0;
        for(int j=0; j<mse.size(); j++){
            avg_mse += mse[j];
        }

        std::printf("Test Average MSE: %f\n", avg_mse);
        // print probabilities
        std::printf("Expected label: %d\n", test_sample.label);
        for(int j=0; j<result[0].size(); j++){
            std::printf("\tProb %d -> %f %% \n", j, result[0][j]*100);
        }
        
        // if(i == epochs-1){
        //     for(int j = 0; j<loss.size(); j++){
        //         std::printf("Loss[%d]: %f\n", j, loss[j]);
        //     }
        // }

        // calculate the loss for this step using mse
    }
}

std::vector<float> Model::mse(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2){
    if(v1.size() != v2.size()){
        std::printf("Error: Sizes of vectors do not match, v1(%ld, X) != v2(%ld, X)\n", v1.size(), v2.size());
        return std::vector<float>();
    }
    std::vector<float> mse(v1[0].size(), 0.0f);
    for(int i=0; i<v1.size(); i++){
        if(v1[i].size() != v2[i].size()){
            std::printf("Error: Sizes of vectors do not match v1(%ld, %ld) != v2(%ld, %ld)\n", v1.size(), v1[i].size(), v2.size(), v2[i].size());
            return std::vector<float>();
        }
        for(int j=0; j<v1[i].size(); j++){
            mse[j] += 0.5*std::pow(v1[i][j] - v2[i][j], 2);
        }
    }

    // average for v1.size()
    for(int i=0; i<mse.size(); i++){
        mse[i] /= static_cast<float>(v1.size());
    }

    return mse;
}

std::vector<float> Model::loss(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2){
    std::vector<float> loss(v1.size(), 0.0f);
    for(int i=0; i<v1.size(); i++){
        for(int j=0; j<v1[i].size(); j++){
            loss[i] += (v1[i][j] - v2[i][j]);
        }
    }

    return loss;
}

std::vector<float> Model::get_one_hot(int label, int size){
    std::vector<float> one_hot(size, 0);
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

