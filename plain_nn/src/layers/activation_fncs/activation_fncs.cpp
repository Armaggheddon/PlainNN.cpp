#include "activation_fncs.hpp"
#include "utils.hpp"

#include <string>


ActivationFn* get_activation_fn_from_name(std::string name){
    ActivationFn* activation_fn;
    std::string activation_name = string_to_lower(name);

    if(activation_name.compare(string_to_lower(ACTIVATION_NAMES[ActivationType::RELU])) == 0){
        activation_fn = new ReLU();
    } else if(activation_name.compare(string_to_lower(ACTIVATION_NAMES[ActivationType::SIGMOID])) == 0){
        activation_fn = new Sigmoid();
    } else if(activation_name.compare(string_to_lower(ACTIVATION_NAMES[ActivationType::TANH])) == 0){
        activation_fn = new Tanh();
        ;
    } else {
        activation_fn = new None();
    }

    return activation_fn;
}