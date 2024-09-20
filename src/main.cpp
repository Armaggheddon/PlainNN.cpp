#include <iostream>
#include <fstream>
#include "model.h"


int main(){

    Model model;
    model.add_layer(new Input({784}));
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));
    
    model.summary();

    MNISTDataLoader train_data_loader(
        "../data/mnist_dataset/train-images-idx3-ubyte", 
        "../data/mnist_dataset/train-labels-idx1-ubyte",
        true, 
        true);
    train_data_loader.load();
    
    model.set_lr_scheduler(new StepLR(0.8, 4));
    model.train(
        train_data_loader,
        0.1,
        20,
        64,
        true,
        "../data/model_save/trained_model");

    MNISTDataLoader test_data_loader("../data/mnist_dataset/t10k-images-idx3-ubyte", "../data/mnist_dataset/t10k-labels-idx1-ubyte", true, true);
    test_data_loader.load();

    EvaluationResult result = model.evaluate(test_data_loader);
    std::printf("Correct: %d/%d\n", result.correct, result.total);

    return 0;
}