#include "model.h"

#include <iostream>

int main(){

    // create a model with 3 layers
    Model model;
    model.add_layer(new Input({784}));
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));

    // Load the MNIST dataset
    MNISTDataLoader data_loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", true, true);
    data_loader.load();

    // Optionally set the learning rate scheduler
    model.set_lr_scheduler(new StepLR(0.8, 1));

    // Print the model summary
    model.summary();

    // Train the model and save the checkpoints to "../data/model_save"
    model.train(data_loader, 0.01, 1, 64, true, "../data/model_save");

    // Load the test dataset for MNIST
    MNISTDataLoader test_data_loader("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", true, true);
    test_data_loader.load();

    // Evaluate the model on the test dataset
    EvaluationResult result = model.evaluate(data_loader);

    std::printf("Correct: %d/%d\n", result.correct, result.total);

    return 0;
}