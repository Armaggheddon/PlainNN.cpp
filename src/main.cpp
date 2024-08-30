#include <cstdio>
#include <bits/stdc++.h>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <cmath>
#include <string> // Add this line to include the <string> header
// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

#include "model.h"
#include "layers.h"
#include "activation.h"
#include "initialization.h"
#include "data_loader.h"

#include "model_loader.h"

int main(int argc, char* argv[]){

    Model loaded_model = ModelLoader::loadJson("../checkpoints/mnist/ckpt_epoch_0.json");
    ModelLoader::loadWeights("../checkpoints/mnist/ckpt_epoch_0.weights", loaded_model);

    for(int i=0; i<argc; i++){
        std::printf("Arg %d: %s\n", i, argv[i]);
    }

    MNISTDataLoader mnist_dataset = MNISTDataLoader("../examples", 0.01, true);
    mnist_dataset.load();
    mnist_dataset.shuffle();


    Model model = Model();

    // model.load("../checkpoints/mnist/ckpt_epoch_0");

    model.add(new Input(784));
    model.add(new Dense(128, new ReLU())); //128, relu
    model.add(new Dense(10, new Softmax())); //10, softmax

    Data input_sample = mnist_dataset.get_sample();

    model.compile();
    model.summary();
    std::vector<std::vector<float> > result = model.forward(new std::vector<std::vector<float> >(1, input_sample.input));
    std::printf("Expected label: %d\n", input_sample.label);
    for(int i=0; i<result.size(); i++){
        std::printf("Batch[%d]:\n", i);
        for(int j=0; j<result[i].size(); j++){
            std::printf("\tProb %d -> %f %% \n", j, result[i][j]*100);
        }
    }

    model.train(&mnist_dataset, 20, 64, 0.2, "../checkpoints/mnist");

    Data sample = mnist_dataset.get_sample();
    std::vector<std::vector<float> > sample_input(1, std::vector<float>(784, 0));
    for(int i=0; i<sample.input.size(); i++){
        sample_input[0][i] = sample.input[i];
    }

    std::vector<std::vector<float> > test = model.forward(&sample_input);
    
    std::printf("Expected label: %d\n", sample.label);
    for(int j=0; j<result[0].size(); j++){
        std::printf("\tProb %d -> %f %% \n", j, result[0][j]*100);
    }
}