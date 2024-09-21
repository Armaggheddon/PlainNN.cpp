#include "plain_nn.hpp"
#include <iostream>

int main(){

    PlainNN saved_model;
    saved_model.load("../test/model_save/model_save_epoch_0");

    saved_model.summary();

    // Train the model for another epoch
    MNISTDataLoader train_dataloader("../test/train-images-idx3-ubyte", "../test/train-labels-idx1-ubyte", true, true);
    train_dataloader.load();

    saved_model.train(train_dataloader, 0.01, 1, 64);

    // Evaluate the model on the test datasets
    MNISTDataLoader test_dataloader("../test/t10k-images-idx3-ubyte", "../test/t10k-labels-idx1-ubyte", true, true);
    test_dataloader.load();
    EvaluationResult res = saved_model.evaluate(train_dataloader);
    std::printf("Correct: %d/%d\n", res.correct, res.total);

    // Save the new model
    saved_model.save("../test/model_save/model_save_epoch_1");

    return 0;
}