#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include "model.h"
#include "model_loader.h"

Model::Model(){
    this->layers = std::vector<Layer*>();
    this->is_initialized = false;
}

void Model::_check_initialized(){
    if(!this->is_initialized){
        std::printf("Error: Model not initialized, call initialize before running forward pass\n");
        throw std::runtime_error("Model not initialized");
    }
}

void Model::add(Layer *layer){
    if(this->is_initialized){
        // Calling add after the model has been initialized 
        // will reset the model to an un-initialized state
        // requiring another call to initialize
        this->is_initialized = false;
    }

    LayerInfo layer_info = layer->get_info();
    if(this->layers.size() == 0 && (layer_info.layer_name.compare("input"))){
        std::printf("Error: First layer must be Input layer\n");
        return;
    }
    if(this->layers.size() == 0) this->input_size = layer_info.output_neurons;
    this->layers.push_back(layer);
}

std::vector<std::vector<float> > Model::forward(std::vector<std::vector<float> > *input){
    this->_check_initialized();
    
    this->layers[1]->forward(input);
    this->layers[2]->forward(&this->layers[1]->output);
    return this->layers[2]->output;
}

void Model::train(DataLoader *x, int epochs, int batch_size, float learning_rate, std::string checkpoints_root, int from_epoch){
    this->_check_initialized();

    Data test_data = x->get_sample();
    std::vector<std::vector<float> > test_input(1, std::vector<float>(784, 0));
    int test_label = test_data.label;
    for(int i=0; i<test_data.input.size(); i++){
        test_input[0][i] = test_data.input[i];
    }

    std::string checkpoint_file_name_template = checkpoints_root + "/ckpt_epoch_%d";

    for(int epoch = from_epoch; epoch<epochs; epoch++){

        x->new_epoch();

        int layer_count = this->layers.size() - 1; // exclude input layer

        std::printf("Epoch %d\n", epoch);

        int steps = std::ceil(static_cast<double>(x->train_data.size())/batch_size);
        for(int step = 0; step < steps; step++){

            std::vector<Data> batch_data = x->get_batch(batch_size);

            // TODO: move this to the layer class
            std::fill(this->layers[1]->weight_gradients.begin(), this->layers[1]->weight_gradients.end(), std::vector<float>(this->layers[1]->output[0].size(), 0));
            std::fill(this->layers[2]->weight_gradients.begin(), this->layers[2]->weight_gradients.end(), std::vector<float>(this->layers[2]->output[0].size(), 0));

            std::vector<std::vector<float> > batch_inputs(batch_size, std::vector<float>(batch_data.size(), 0));
            
            for(int j=0; j<batch_data.size(); j++){
                batch_inputs[j] = batch_data[j].input;
            }

            std::vector<std::vector<float> > result = this->forward(&batch_inputs);

            std::vector<std::vector<float> > target(batch_size, std::vector<float>(10, 0));
            for(int j=0; j<batch_data.size(); j++){
                target[j] = this->_get_one_hot(batch_data[j].label, 10);
            }
            std::vector<float> mse = this->mse(result, target);

            float tot_mse = 0;
            for(int j=0; j<mse.size(); j++){
                tot_mse += mse[j];
            }

            
            // calculate delta for last layer
            for(int batch = 0; batch<result.size(); batch++){
                for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
                    for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                        this->layers[2]->weight_gradients[perceptron][weight] += (target[batch][perceptron] - result[batch][perceptron]) * result[batch][perceptron] * (1 - result[batch][perceptron]) * this->layers[1]->output[batch][weight];
                    }
                }
            }

            // calculate delta for second last layer
            for(int batch = 0; batch<result.size(); batch++){
                for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
                    for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                        float eo1_net01 = 0;
                        for(int perceptron2 = 0; perceptron2 < this->layers[2]->weights.size(); perceptron2++){
                            eo1_net01 += (target[batch][perceptron2] - result[batch][perceptron2]) * result[batch][perceptron2] * (1 - result[batch][perceptron2]) * this->layers[2]->weights[perceptron2][perceptron];
                        }
                        this->layers[1]->weight_gradients[perceptron][weight] += eo1_net01 * 1 * batch_inputs[batch][weight];
                    }
                }
            }
                        
            char progress_bar[20+1] = {0};
            int progress = (int)((static_cast<float>(step+1)/steps)*20); 
            for(int i=0; i<progress; i++){
                progress_bar[i] = '=';
            }
            for(int i=progress; i<20; i++){
                progress_bar[i] = ' ';
            }
            progress_bar[20] = '\0';
            std::printf("\r\t|%s| %d/%d Step loss: %f", 
                progress_bar, step+1, steps, tot_mse);
            std::fflush(stdout);
        }

        std::printf("\n");

        // update layers weights
        for(int perceptron = 0; perceptron < this->layers[2]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[2]->weights[perceptron].size(); weight++){
                // average the gradient for the minibatch, and for the results 
                this->layers[2]->weights[perceptron][weight] += (learning_rate * this->layers[2]->weight_gradients[perceptron][weight]) / (batch_size * steps);
            }
        }
        // update weights for second last layer
        for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                this->layers[1]->weights[perceptron][weight] += (learning_rate * this->layers[1]->weight_gradients[perceptron][weight]) / (batch_size * steps);
            }
        }

        if(checkpoints_root != ""){
            char *checkpoint_file_name = new char[checkpoint_file_name_template.size() + 10];
            std::sprintf(checkpoint_file_name, checkpoint_file_name_template.c_str(), epoch);
            this->save(checkpoint_file_name);
        }

        // run forward pass
        std::vector<std::vector<float> > result = this->forward(&test_input);
        std::vector<std::vector<float> > target;
        target.push_back(this->_get_one_hot(test_data.label, 10));
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

std::vector<float> Model::_get_one_hot(int label, int size){
    std::vector<float> one_hot(size, 0);
    one_hot[label] = 1;
    return one_hot;
}

void Model::save(std::string filename){
    // The output consists of 2 files, a model description file
    // and a model weights file, the model description file is a json file
    // with the layers and their parameters, the model weights file is a binary file
    // filename.json and filename.weights
    
    std::string model_description_filename = filename + ".json";
    std::string model_weights_filename = filename + ".weights";

    FILE *model_description_file = std::fopen(model_description_filename.c_str(), "w+");
    if(model_description_file == nullptr){
        std::printf("Error: Could not open file %s\n", model_description_filename.c_str());
        return;
    }

    // The general structure is a json object with a layers array
    // each layer is a json object with a layer name and a parameters object
    // the parameters object contains the parameters of the layer
    std::string model_description = "{";
    char tmp_str_buffer[4096] = {0};
    int layer_id = 0;
    for(Layer *layer : this->layers){
        
        LayerInfo info = layer->get_info();

        std::sprintf(
            tmp_str_buffer,
            "\"%s_%d\": {"
                "\"layer_name\": \"%s\","
                "\"activation_fn_name\": \"%s\","
                "\"input_neurons\": %d,"
                "\"output_neurons\": %d,"
                "\"param_count\": %ld,"
                "\"parameters\": {"
                    "\"weights_count\": %d,"
                    "\"weights_shape\": [%d, %d],"
                    "\"weight_bytes_count\": %ld,"
                    "\"biases_count\": %d,"
                    "\"biases_shape\": [%d],"
                    "\"bias_bytes_count\": %ld"
                "}"
            "},", 
            info.layer_name.c_str(), layer_id++, 
            info.layer_name.c_str(), 
            info.activation_fn_name.c_str(),
            info.input_neurons, 
            info.output_neurons,
            info.param_count,
            info.weights_count,
            info.output_neurons, info.input_neurons,
            info.weight_bytes_count,
            info.biases_count,
            info.output_neurons,
            info.bias_bytes_count
            );
        model_description += tmp_str_buffer;
    }

    model_description[model_description.size()-1] = '}';

    std::fprintf(model_description_file, "%s", model_description.c_str());
    std::fclose(model_description_file);

    // write weights to binary file
    FILE *model_weights_file = std::fopen(model_weights_filename.c_str(), "wb+");
    if(model_weights_file == nullptr){
        std::printf("Error: Could not open file %s\n", model_weights_filename.c_str());
        return;
    }

    for(Layer *layer : this->layers){
        for(int i=0; i<layer->weights.size(); i++){
            std::fwrite(layer->weights[i].data(), sizeof(float), layer->weights[i].size(), model_weights_file);
        }
        
        std::fwrite(layer->biases.data(), sizeof(float), layer->biases.size(), model_weights_file);
        
    }

    std::fclose(model_weights_file);
}


void Model::summary(){
    std::printf("%-10s%-10s%-10s\n", "Layer", "Output", "Param #");
    std::printf("------------------------------------------\n");
    float trainable_param_count = 0;
    for(int i=0; i<this->layers.size(); i++){
        LayerInfo info = this->layers[i]->get_info();
        std::string layer_name = info.layer_name + "_" + std::to_string(i);
        std::string output_size = "( , " + std::to_string(info.output_neurons) + ")";
        std::printf("%-10s%-10s%ld\n", layer_name.c_str(), output_size.c_str(), info.param_count);
        trainable_param_count += info.param_count;
    }
    std::printf("------------------------------------------\n");
    std::printf("Total trainable parameters: %d\n", (int)trainable_param_count);
}

void Model::initialize(){
    this->layers[1]->initialize(this->input_size);
    for(int i=2; i<this->layers.size(); i++){
        this->layers[i]->initialize(this->layers[i-1]->output[0].size());
    }

    this->layers_count = this->layers.size();
    this->is_initialized = true;
}

Layer* Model::operator[](int index){
    if(index >= this->layers.size()){
        std::printf("Error: Index out of bounds, index %d, layers count %ld\n", index, this->layers.size());
        throw std::out_of_range("Index out of bounds");
    }
    return this->layers[index];
}

Model Model::from_json(std::string filename){

    if (filename.substr(filename.find_last_of(".")).compare(".json") != 0)
    {
        std::printf("Error: Invalid file format, expected .json file\n");
        throw std::invalid_argument("Invalid file format, expected .json file");
    }
    
    Model model = ModelLoader::loadJson(filename);
    model.initialize();
    return model;
}

Model Model::from_checkpoint(std::string base_filename){

    int idx = base_filename.find_last_of(".");
    if(idx != std::string::npos){
        // means that has at least one "." in the filename
        std::string base_filename_ext = base_filename.substr(idx);
        if(base_filename_ext.compare(".json") == 0 || base_filename_ext.compare(".weights") == 0){
            throw std::invalid_argument("Invalid file format, expected base filename without extension, got: " + base_filename);
        }
    }

    std::string json_filename = base_filename + ".json";
    std::string weights_filename = base_filename + ".weights";

    Model model = Model::from_checkpoint(json_filename, weights_filename);
    return model;
}

Model Model::from_checkpoint(const std::string json_filename, const std::string weights_filename){

    if(json_filename.substr(json_filename.find_last_of(".")).compare(".json") != 0){
        std::printf("Error: Invalid file format, expected .json file\n");
        throw std::invalid_argument("Invalid file format, expected .json file");
    }
    if(weights_filename.substr(weights_filename.find_last_of(".")).compare(".weights") != 0)
    {
        std::printf("Error: Invalid file format, expected .weights file\n");
        throw std::invalid_argument("Invalid file format, expected .weights file");
    }

    Model model = ModelLoader::loadJson(json_filename);
    ModelLoader::loadWeights(weights_filename, model);
    model.initialize();
    return model;
}