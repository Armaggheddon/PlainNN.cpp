#include "model.h"

int main(){

    // create a model with 3 layers
    Model model;
    model.add_layer(new Input({784}));
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));

    // Print the model summary
    model.summary();

    return 0;
}