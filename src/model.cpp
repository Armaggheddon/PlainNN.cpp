#include <iostream>
#include <fstream>
#include <map>
#include <string>
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

void Model::train(DataLoader *x, int epochs, int batch_size, float learning_rate, std::string checkpoints_path){
    Data test_data = x->get_sample();
    std::vector<std::vector<float> > test_input(1, std::vector<float>(784, 0));
    int test_label = test_data.label;
    for(int i=0; i<test_data.input.size(); i++){
        test_input[0][i] = test_data.input[i];
    }

    std::string checkpoint_file_name_template = checkpoints_path + "/ckpt_epoch_%d";

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
                        this->layers[2]->grad[perceptron][weight] += (target[batch][perceptron] - result[batch][perceptron]) * result[batch][perceptron] * (1 - result[batch][perceptron]) * this->layers[1]->output[batch][weight];
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
                        this->layers[1]->grad[perceptron][weight] += eo1_net01 * 1 * batch_inputs[batch][weight];
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
                this->layers[2]->weights[perceptron][weight] += (learning_rate * this->layers[2]->grad[perceptron][weight]) / (batch_size * steps);
            }
        }
        // update weights for second last layer
        for(int perceptron = 0; perceptron < this->layers[1]->weights.size(); perceptron++){
            for(int weight = 0; weight < this->layers[1]->weights[perceptron].size(); weight++){
                this->layers[1]->weights[perceptron][weight] += (learning_rate * this->layers[1]->grad[perceptron][weight]) / (batch_size * steps);
            }
        }

        if(checkpoints_path != ""){
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
    if(model_description_file == NULL){
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
        
        LayerSummary s = layer->get_summary();

        std::sprintf(
            tmp_str_buffer,
            "\"%s_%d\": {"
                "\"layer_name\": \"%s\","
                "\"activation_fn\": \"%s\","
                "\"output_size\": %d,"
                "\"param_count\": %d,"
                "\"parameters\": {"
                    "\"weights\": [%ld, %ld],"
                    "\"weights_bytes\": %ld,"
                    "\"biases\": %d,"
                    "\"biases_bytes\": %d"
                "}"
            "},", 
            s.layer_name.c_str(), layer_id++, 
            s.layer_name.c_str(), 
            s.activation_fn.c_str(),
            s.output_size, 
            s.param_count, 
            (!s.layer_name.compare("input")) ? 0 : layer->weights.size(), (!s.layer_name.compare("input")) ? 0 : layer->weights[0].size(), 
            (!s.layer_name.compare("input")) ? 0 : layer->weights.size() * layer->weights[0].size() * sizeof(float),
            0,
            0);
        model_description += tmp_str_buffer;
    }

    model_description[model_description.size()-1] = '}';

    std::fprintf(model_description_file, "%s", model_description.c_str());
    std::fclose(model_description_file);

    // write weights to binary file
    FILE *model_weights_file = std::fopen(model_weights_filename.c_str(), "wb+");
    if(model_weights_file == NULL){
        std::printf("Error: Could not open file %s\n", model_weights_filename.c_str());
        return;
    }

    for(Layer *layer : this->layers){
        for(int i=0; i<layer->weights.size(); i++){
            std::fwrite(layer->weights[i].data(), sizeof(float), layer->weights[i].size(), model_weights_file);
        }
        // TODO write biases too
    }

    std::fclose(model_weights_file);
}

void Model::load(std::string filename){

    // check if filename has any extension, if is the case
    // load only the extension, otherwise load both .json and .weights
    // files

    bool has_extension = false;
    std::size_t is_json = filename.find(".json") != std::string::npos; // if index is not found with find, returns std::string::npos
    std::size_t is_weights = filename.find(".weights") != std::string::npos;
    std::printf("Loading model from %s\n", filename.c_str());
    if( is_json || is_weights){
        if(is_json){
            // parse only json file
            std::vector<LayerSummary> layers = this->_parse_json(filename);
            this->_build_model_from_layer_summary(layers);
            std::printf("Parsing json file: %s\n", filename.c_str());
        }
        else if (is_weights){
            // parse only weights file
            if(this->layers.size() == 0){
                std::printf("Error: Model has no layers\n");
                return;
            }
            this->_parse_weights(filename);
            std::printf("Parsing weights file: %s\n", filename.c_str());
        }
        else{
            std::printf("Error: Invalid file extension\n");
            return;
        }
    }
    else{
        // parse both files
        std::string json_filename = filename + ".json";
        std::string weights_filename = filename + ".weights";
        std::vector<LayerSummary> layers = this->_parse_json(json_filename);
        std::printf("Parsing json file: %s\n", json_filename.c_str());
        this->_build_model_from_layer_summary(layers);
        this->_parse_weights(weights_filename);
        std::printf("Parsing weights file: %s\n", weights_filename.c_str());
    }

    std::printf("Model loaded\n");
    summary();
}

std::vector<LayerSummary> Model::_parse_json(std::string filename){
    std::ifstream model_description_file(filename);
    if(!model_description_file.is_open()){
        std::printf("Error: Could not open file %s\n", filename.c_str());
        return std::vector<LayerSummary>();
    }

    std::string line;
    std::string model_description;
    while(std::getline(model_description_file, line)){
        model_description += line;
    }

    model_description_file.close();
    
    std::vector<LayerSummary> layers;
    return layers;
}

void Model::_parse_weights(std::string filename){
    FILE *weights_file = std::fopen(filename.c_str(), "rb");
    if(weights_file == NULL){
        std::printf("Error: Could not open file %s\n", filename.c_str());
        return;
    }

    for(float val = 0; std::fread(&val, sizeof(float), 1, weights_file) == 1;){
        std::printf("%f\n", val);
    }

    std::fclose(weights_file);
}

void Model::_build_model_from_layer_summary(std::vector<LayerSummary> layer_summary){

    for(int i=0; i<layer_summary.size(); i++){
        LayerSummary s = layer_summary[i];
        if(s.layer_name.compare("input") == 0){
            std::printf("Adding input layer\n");
            this->add(new Input(s.output_size));
        }
        else if(s.layer_name.compare("dense") == 0){
            std::printf("Adding dense layer\n");
            this->add(new Dense(s.output_size, new ReLU()));
        }
        else if(s.layer_name.compare("softmax") == 0){
            std::printf("Adding softmax layer\n");
            this->add(new Dense(s.output_size, new Softmax()));
        }
    }

    this->compile();

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

    this->layers_count = this->layers.size();
}

Layer* Model::operator[](int index){
    return this->layers[index];
}

