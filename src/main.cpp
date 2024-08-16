#include <cstdio>
#include <bits/stdc++.h>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <cmath>
#include <string> // Add this line to include the <string> header
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "model.h"
#include "layers.h"
#include "activation.h"
#include "initialization.h"

int main(int argc, char* argv[]){

    for(int i=0; i<argc; i++){
        std::printf("Arg %d: %s\n", i, argv[i]);
    }

    int width = 0, height = 0, channels = 0;
    unsigned char *data = stbi_load("examples/bg.jpg", &width, &height, &channels, 3);

    Model model = Model();
    model.add(new Input(784));
    model.add(new Dense(128, &relu));
    model.add(new Dense(10, &softmax));

    std::vector<std::vector<float> > input(1, std::vector<float>(784, 0));
    random_uniform(&input, 0, 1);

    model.compile();
    model.summary();
    std::vector<std::vector<float> > result = model.forward(&input);

    for(int i=0; i<result.size(); i++){
        std::printf("Batch[%d]:\n", i);
        for(int j=0; j<result[i].size(); j++){
            std::printf("\tProb %d -> %f %% \n", j, result[i][j]*100);
        }
    }
}