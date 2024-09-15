#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <fstream>
#include "model.h"
#include "layers.h"
#include "activation_fncs.h"
#include "data_loaders.h"


int main(){

    Model model;
    model.add_layer(new Input({784}));
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));

    MNISTDataLoader train_dataloader("../test/train-images-idx3-ubyte", "../test/train-labels-idx1-ubyte", true, true);
    train_dataloader.load();
    
    MNISTDataLoader test_dataloader("../test/t10k-images-idx3-ubyte", "../test/t10k-labels-idx1-ubyte", true, true);
    test_dataloader.load();
    
    // model.load("../test/model_save");
    model.summary();
    // EvaluationResult res = model.evaluate(test_dataloader);

    // std::printf("Correct: %d/%d\n", res.correct, res.total);
    
    model.train(train_dataloader, test_dataloader, 0.01, 1, 64);

    // model.save("../test/model_save");


    return 0;
}