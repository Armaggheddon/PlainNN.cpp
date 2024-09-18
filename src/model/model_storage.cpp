#include "model_storage.h"

#include "model/layers/layers.h"
#include "json.h"
#include <fstream>


void ModelStorage::save_model_arch(
    std::string file_name,
    std::vector<LayerSummary> layer_summaries
){
    std::ofstream file(file_name + MODEL_ARCH_FILE_EXT);

    file << "{\n";

    for(int summary_idx = 0; summary_idx < layer_summaries.size(); summary_idx++){
        LayerSummary summary = layer_summaries[summary_idx];

        file << "    \"" << std::to_string(summary_idx) << "\": {\n";
        file << "        \"layer_name\": \"" << summary.layer_name << "\",\n";
        file << "        \"activation_fn\": \"" << summary.activation_fn << "\",\n";
        file << "        \"param_count\": " << summary.param_count << ",\n";
        file << "        \"param_size\": " << summary.param_size << ",\n";
        file << "        \"layer_shape\": [";

        for(int shape_idx = 0; shape_idx < summary.layer_shape.size(); shape_idx++){
            file << summary.layer_shape[shape_idx];
            if(shape_idx < summary.layer_shape.size() - 1){
                file << ", ";
            }
        }
        file << "]\n";
        file << "    }";
        if(summary_idx < layer_summaries.size() - 1){
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";

    file.close();
}

void ModelStorage::save_model_weights(
    std::string file_name,
    std::vector<std::vector<double> > weights
){
    std::ofstream file(file_name + MODEL_WEIGHTS_FILE_EXT, std::ios::binary);

    for(int weight_idx = 0; weight_idx < weights.size(); weight_idx++){
        std::vector<double> weight = weights[weight_idx];

        file.write((char*)weight.data(), weight.size() * sizeof(double));
    
    }

    file.close();

}


void ModelStorage::load_model_arch(
    std::string file_name,
    Model& model
){
    std::ifstream file(file_name + MODEL_ARCH_FILE_EXT);
    std::string line, lines;

    while(std::getline(file, line)){
        lines += line;
    }
    file.close();

    struct json_value_s* root = json_parse(lines.c_str(), lines.size());
    if(root == NULL){
        throw std::runtime_error("Error parsing JSON file.");
    }

    struct json_object_s* root_obj = json_value_as_object(root);
    if(root_obj == NULL){
        throw std::runtime_error("Error parsing JSON file.");
    }

    for(json_object_element_s *element = root_obj->start; element != nullptr; element = element->next){

        json_string_s* key = element->name;

        json_object_s* layer_obj = json_value_as_object(element->value);
        
        json_object_element_s* layer_name_obj = layer_obj->start;
        json_string_s* layer_name = json_value_as_string(layer_name_obj->value);

        json_object_element_s* activation_fn_obj = layer_name_obj->next;
        json_string_s* activation_fn = json_value_as_string(activation_fn_obj->value);

        json_object_element_s* param_count_obj = activation_fn_obj->next;
        json_number_s* param_count = json_value_as_number(param_count_obj->value);

        json_object_element_s* param_size_obj = param_count_obj->next;
        json_number_s* param_size = json_value_as_number(param_size_obj->value);

        json_object_element_s* layer_shape_obj = param_size_obj->next;
        json_array_s* layer_shape = json_value_as_array(layer_shape_obj->value);

        std::vector<int> shape;
        for(json_array_element_s* shape_element = layer_shape->start; shape_element != nullptr; shape_element = shape_element->next){
            json_number_s* shape_num = json_value_as_number(shape_element->value);
            shape.push_back(std::stoi(shape_num->number));
        }

        std::string activation_fn_str = std::string(activation_fn->string);
        ActivationFn* activation_fn_ptr = get_activation_fn_from_name(activation_fn_str);
        
        std::string layer_name_str = std::string(layer_name->string);
        Layer* layer = build_layer_from_name(layer_name_str, shape, activation_fn_ptr);
        
        model.add_layer(layer);
    }

    // TODO: Free the JSON object
}


void ModelStorage::load_model_weights(
    std::string file_name,
    int layer_count,
    Model& model
){
    std::ifstream file(file_name + MODEL_WEIGHTS_FILE_EXT, std::ios::binary);

    for(int layer_idx = 0; layer_idx < layer_count; layer_idx++){
        Layer* layer = model.get_layer(layer_idx);

        LayerSummary summary = layer->get_summary();

        int params_count = summary.param_count;

        std::vector<double> weights(params_count, 0);

        file.read((char*)weights.data(), params_count * summary.param_size);

        layer->load_params(weights);
    }

    file.close();
}