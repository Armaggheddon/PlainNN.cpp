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
    Data test_data = x->get_sample();
    std::vector<std::vector<float> > test_input(1, std::vector<float>(784, 0));
    int test_label = test_data.label;
    for(int i=0; i<test_data.input.size(); i++){
        test_input[0][i] = test_data.input[i];
    }

    for(int epoch = 0; epoch<epochs; epoch++){

        x->new_epoch();

        int layer_count = this->layers.size() - 1; // exclude input layer

        std::printf("Epoch %d\n", epoch);

        int steps = std::ceil(static_cast<double>(x->train_data.size())/batch_size);
        for(int step = 0; step < steps; step++){

            std::vector<Data> batch_data = x->get_batch(batch_size);

            // TODO: move this to the layer class
            std::fill(this->layers[1]->grad.begin(), this->layers[1]->grad.end(), std::vector<float>(this->layers[1]->output[0].size(), 0));
            std::fill(this->layers[2]->grad.begin(), this->layers[2]->grad.end(), std::vector<float>(this->layers[2]->output[0].size(), 0));

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
            
            // calculate delta for last layer
            for(int batch = 0; batch<result.size(); batch++){
                for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
                    for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                        this->layers[2]->grad[perceptron][weight] += (target[batch][perceptron] - result[batch][perceptron]) * result[batch][perceptron] * (1 - result[batch][perceptron]) * this->layers[1]->output[batch][weight];
                    }
                }
            }

            // calculate delta for second last layer
            for(int batch = 0; batch<result.size(); batch++){
                for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
                    for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                        float eo1_net01 = 0;
                        for(int i=0; i<this->layers[2]->weights.size(); i++){
                            eo1_net01 += (target[batch][i] - result[batch][i]) * this->layers[2]->output[batch][i] * (1 - this->layers[2]->output[batch][i]);
                        }
                        this->layers[1]->grad[perceptron][weight] += eo1_net01 * 1 * this->layers[1]->weights[perceptron][weight] * batch_inputs[batch][weight];
                    }
                }
            }
                        
            char progress_bar[20+1] = {0};
            int progress = (int)((static_cast<float>(step)/steps)*20);
            for(int i=0; i<progress; i++){
                progress_bar[i] = '=';
            }
            for(int i=progress; i<20; i++){
                progress_bar[i] = ' ';
            }
            progress_bar[20] = '\0';
            std::printf("\r\t|%s| %d/%d Loss: %f", 
                progress_bar, step, steps, avg_mse);
            std::fflush(stdout);
        }

        std::printf("\n");

        // update layers weights
        for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                // average the gradient for the minibatch, and for the results 
                this->layers[2]->weights[perceptron][weight] += (learning_rate * this->layers[2]->grad[perceptron][weight]) / (batch_size * steps);
            }
        }
        // update weights for second last layer
        for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                this->layers[1]->weights[perceptron][weight] += (learning_rate * this->layers[1]->grad[perceptron][weight]) / (batch_size * steps);
            }
        }

        // run forward pass
        std::vector<std::vector<float> > result = this->forward(&test_input);
        std::vector<std::vector<float> > target;
        target.push_back(this->get_one_hot(test_data.label, 10));
        std::vector<float> mse = this->mse(result, target);

        float avg_mse = 0;
        for(int j=0; j<mse.size(); j++){
            avg_mse += mse[j];
        }

        std::printf("\tTest Average MSE: %f\n", avg_mse);
        // print probabilities
        std::printf("\tExpected label: %d\n", test_label);
        for(int j=0; j<result[0].size(); j++){
            std::printf("\t\tProb %d -> %f %% \n", j, result[0][j]*100);
        }
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

