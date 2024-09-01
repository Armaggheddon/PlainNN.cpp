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

        json_object_element_s *input_neurons = activation_fn->next;
        json_number_s* input_neurons_value = json_value_as_number(input_neurons->value);

        json_object_element_s *output_neurons = input_neurons->next;
        json_number_s* output_neurons_value = json_value_as_number(output_neurons->value);

        json_object_element_s *param_count = output_neurons->next;
        json_number_s* param_count_value = json_value_as_number(param_count->value);

        json_object_s *parameters = json_value_as_object(param_count->next->value);

        json_object_element_s *weights_count = parameters->start;
        json_number_s* weights_count_value = json_value_as_number(weights_count->value);

        json_object_element_s *weights_shape = weights_count->next;
        json_array_s* weights_shape_value = json_value_as_array(weights_shape->value);

        json_object_element_s *weight_bytes_count = weights_shape->next;
        json_number_s* weight_bytes_count_value = json_value_as_number(weight_bytes_count->value);

        json_object_element_s *biases_count = weight_bytes_count->next;
        json_number_s* biases_count_value = json_value_as_number(biases_count->value);

        json_object_element_s *biases_shape = biases_count->next;
        json_array_s* biases_shape_value = json_value_as_array(biases_shape->value);

        json_object_element_s *bias_bytes_count = biases_shape->next;
        json_number_s* bias_bytes_count_value = json_value_as_number(bias_bytes_count->value);

        int output_size_int = std::stoi(output_neurons_value->number);
        
        ActivationFn* activation = nullptr;

        std::string log_activation_fn, log_layer_name;

        std::string activation_fn_str = std::string(activation_fn_value->string);
        if(activation_fn_str.compare(ACTIVATION_FN_NAMES[ActivationFnType::RELU]) == 0){
            log_activation_fn = ACTIVATION_FN_NAMES[ActivationFnType::RELU];
            activation = new ReLU();
        }else if(activation_fn_str.compare(ACTIVATION_FN_NAMES[ActivationFnType::SOFTMAX]) == 0){
            log_activation_fn = ACTIVATION_FN_NAMES[ActivationFnType::SOFTMAX];
            activation = new Softmax();
        }else if (activation_fn_str.compare(ACTIVATION_FN_NAMES[ActivationFnType::SIGMOID]) == 0){
            log_activation_fn = ACTIVATION_FN_NAMES[ActivationFnType::SIGMOID];
            activation = new Sigmoid();
        }else{
            ;
        }

        Layer* layer = nullptr;
        std::string layer_name_str = std::string(layer_name_value->string);
        if(layer_name_str.compare(LAYER_NAMES[LayerType::DENSE]) == 0){
            log_layer_name = LAYER_NAMES[LayerType::DENSE];
            layer = new Dense(output_size_int, activation);
        } else if(layer_name_str.compare(LAYER_NAMES[LayerType::INPUT]) == 0){
            log_layer_name = LAYER_NAMES[LayerType::INPUT];
            layer = new Input(output_size_int);
        }else{
            ;
        }

        // Capitalize first letter of activation fn and layer
        log_activation_fn[0] = std::toupper(log_activation_fn[0]);
        log_layer_name[0] = std::toupper(log_layer_name[0]);

        std::printf("Loaded layer %s", log_layer_name.c_str());
        if(activation != nullptr){
            std::printf(" with activation %s", log_activation_fn.c_str());
        }
        std::printf(" with output size %d\n", output_size_int);

        model.add(layer);
    }

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

        LayerInfo info =  layer->get_info();

        int weights_size = info.param_count;

        char float_buff[sizeof(float)];
        char biases_buff[sizeof(float) * info.output_neurons];
        for(int out = 0; out < info.output_neurons; out++){
            for(int in = 0; in < info.input_neurons; in++){
                file.read(float_buff, sizeof(float_buff));
                float weight;
                memcpy(&weight, float_buff, sizeof(float_buff));
                layer->weights[out][in] = weight;
            }
        }
        std::printf("Loaded weights for layer %s: (%d, %d)\n", 
            info.layer_name.c_str(), 
            info.input_neurons, 
            info.output_neurons);

        file.read(biases_buff, sizeof(biases_buff));
        memcpy(layer->biases.data(), biases_buff, sizeof(biases_buff));

        std::printf("Loaded biases for layer %s: (%d)\n", info.layer_name.c_str(), info.output_neurons);
    }

}