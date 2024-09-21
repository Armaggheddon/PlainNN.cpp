#include "layers.hpp"
#include "utils.hpp"

#include <string>
#include <vector>

void Layer::freeze(bool freeze){
    is_frozen = freeze;
}

std::string Layer::name(){
    return LAYER_TYPE_NAMES[layer_type];
}


Layer* build_layer_from_name(std::string name, std::vector<int> layer_shape, ActivationFn* activation_fn){
    Layer* layer;
    std::string layer_name = string_to_lower(name);

    if(layer_name.compare(string_to_lower(LAYER_TYPE_NAMES[LayerType::DENSE])) == 0){
        layer = new Dense(layer_shape[0], layer_shape[1], activation_fn);
    } else if(layer_name.compare(string_to_lower(LAYER_TYPE_NAMES[LayerType::INPUT])) == 0){
        layer = new Input({layer_shape[0]});
    } else {
        std::printf("Layer type not found: %s\n", name.c_str());
        exit(1);
    }

    return layer;
}