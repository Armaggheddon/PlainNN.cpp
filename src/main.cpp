#include <iostream>
#include <fstream>
#include "model.h"


int main(){

    Model model;
    model.add_layer(new Input({784}));
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));

    model.load("../data/utils/mnist_fc128_relu_fc10_sigmoid", true);
    
    model.summary();

    MNISTDataLoader test_data_loader("../data/mnist_dataset/t10k-images-idx3-ubyte", "../data/mnist_dataset/t10k-labels-idx1-ubyte", true, true);
    test_data_loader.load();

    EvaluationResult result = model.evaluate(test_data_loader);
    std::printf("Correct: %d/%d\n", result.correct, result.total);

    return 0;
}