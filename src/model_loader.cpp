#include "model_loader.h"
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "model.h"
#include "layers.h"
#include "activation.h"
#include "json.h"

Model ModelLoader::loadJson(const std::string& path){
    
    // check if path has an extension

    // read the file lines
    std::ifstream file(path);
    std::string lines, line;
    while (std::getline(file, line)) {
        lines += line;
    }

    // parse the json
    struct json_value_s* root = json_parse(lines.c_str(), lines.size());
    if (root == nullptr) {
        throw std::runtime_error("Failed to parse json");
    }

    Model model = Model();

    // iterate over the json object
    struct json_object_s* object = json_value_as_object(root);
    if(root == nullptr){
        throw std::runtime_error("Failed to parse json");
    }

    for( json_object_element_s *element = object->start; element != nullptr ; element = element->next){
        
        json_string_s* key = element->name;
        json_value_s* value = element->value;

        json_object_s* layer_info = json_value_as_object(value);
        json_object_element_s *layer_name = layer_info->start;
        json_string_s* layer_name_value = json_value_as_string(layer_name->value);
        
        json_object_element_s *activation_fn = layer_name->next;
        json_string_s* activation_fn_value = json_value_as_string(activation_fn->value);

        json_object_element_s *output_size = activation_fn->next;
        json_number_s* output_size_value = json_value_as_number(output_size->value);

        json_object_element_s *param_count = output_size->next;
        json_number_s* param_count_value = json_value_as_number(param_count->value);

        json_object_s *parameters = json_value_as_object(param_count->next->value);
        json_object_element_s *weights = parameters->start;
        json_array_s* weights_value = json_value_as_array(weights->value);
        
        json_object_element_s *weight_bytes = weights->next;
        json_number_s* weight_bytes_value = json_value_as_number(weight_bytes->value);

        json_object_element_s *biases = weight_bytes->next;
        json_array_s* biases_value = json_value_as_array(biases->value);

        json_object_element_s *bias_bytes = biases->next;
        json_number_s* bias_bytes_value = json_value_as_number(bias_bytes->value);


        int output_size_int = std::stoi(output_size_value->number);
        
        ActivationFn* activation = nullptr;

        std::string log_activation_fn, log_layer_name;

        std::string activation_fn_str = std::string(activation_fn_value->string);
        if(activation_fn_str.compare("relu") == 0){
            log_activation_fn = "ReLU";
            activation = new ReLU();
        }else if(activation_fn_str.compare("softmax") == 0){
            log_activation_fn = "Softmax";
            activation = new Softmax();
        }else if (activation_fn_str.compare("sigmoid") == 0){
            log_activation_fn = "Sigmoid";
            activation = new Sigmoid();
        }else{
            ;
        }

        Layer* layer = nullptr;
        std::string layer_name_str = std::string(layer_name_value->string);
        if(layer_name_str.compare("dense") == 0){
            log_layer_name = "Dense";
            layer = new Dense(output_size_int, activation);
        } else if(layer_name_str.compare("input") == 0){
            log_layer_name = "Input";
            layer = new Input(output_size_int);
        }else{
            ;
        }

        std::printf("Loaded layer %s", log_layer_name.c_str());
        if(activation != nullptr){
            std::printf(" with activation %s", log_activation_fn.c_str());
        }
        std::printf(" with output size %d\n", output_size_int);

        model.add(layer);

    }

    model.compile();

    return model;
}

void ModelLoader::loadWeights(const std::string& path, Model& model){
    // check if path has an extension

    // read the binary file parsing the weights and biases for each layer
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file");
    }

    for(int i=1; i<model.layers_count; i++){
        Layer* layer = model[i];

        LayerSummary summary =  layer->get_summary();

        int weights_size = summary.param_size;

        char float_buff[sizeof(float)];
        for(int out = 0; out < summary.output_size; out++){
            for(int in = 0; in < summary.input_size; in++){
                file.read(float_buff, sizeof(float_buff));
                float weight;
                memcpy(&weight, float_buff, sizeof(float_buff));
                layer->weights[out][in] = weight;
            }
        }
        std::printf("Loaded weights for layer %s: (%d, %d)\n", summary.layer_name.c_str(), summary.input_size, summary.output_size);

        // load biases
        for(int out = 0; out < summary.output_size; out++){
            file.read(float_buff, sizeof(float_buff));
            float bias;
            memcpy(&bias, float_buff, sizeof(float_buff));
            // layer->bias[out] = bias; // TODO: allow bias setting
        }

        std::printf("Loaded biases for layer %s: (%d)\n", summary.layer_name.c_str(), summary.output_size);
    }

}