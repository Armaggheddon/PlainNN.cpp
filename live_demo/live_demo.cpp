#include <iostream>
#include <vector>
#include <cstdio>
#include "plain_nn.h"
#include "utils.h"

int main(int argc, char *argv[]){

    if(argc != 2){
        std::cerr << "No arguments given. Expected usage: live_demo <model_path>" << std::endl;
        throw std::invalid_argument("No arguments given. Expected usage: live_demo <model_path>");
    }

    std::string model_path = argv[1];

    PlainNN model;
    model.load(model_path);


    //size_t input_size = 784 * sizeof(double), output_size = 10 * sizeof(double);
    size_t output_count = 10;
    
    while(1){
        
        // Read the input data
        size_t size = 0;
        std::cin.read(reinterpret_cast<char*>(&size), sizeof(size_t)); // Read vector size
        std::vector<double> input_vector(784);
        std::cin.read(reinterpret_cast<char*>(input_vector.data()), size * sizeof(double)); // Read vector data

        Tensor input_tensor({784}, input_vector);

        Tensor output = model.forward(input_tensor);
        // Tensor softmax_output = softmax(output);

        // Write the output data
        std::cout.write(reinterpret_cast<const char*>(&output_count), sizeof(size_t)); // Write vector size
        std::cout.write(reinterpret_cast<const char*>(output.data()), 10 * sizeof(double)); // Write vector data
        std::cout.flush();
    }
    
    return 0;
}
