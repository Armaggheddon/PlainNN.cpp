#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include "initialization.h"

void random_uniform(std::vector<std::vector<float> > *matrix, float min, float max){
    srand((unsigned int) time(nullptr));
    for(int i=0; i<matrix->size(); i++){
        for(int j=0; j<matrix->at(i).size(); j++){
            matrix->at(i).at(j) = min + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(max-min)));
        }    
    }
}

void random_normal(std::vector<std::vector<float> > *matrix, float mean, float std){
    srand((unsigned int) time(nullptr));
    for(int i=0; i<matrix->size(); i++){
        for(int j=0; j<matrix->at(i).size(); j++){
            matrix->at(i).at(j) = mean + std * static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX));
        }
    }
}

void zeros(std::vector<std::vector<float> > *matrix){
    constant(matrix, 0);
}

void ones(std::vector<std::vector<float> > *matrix){
    constant(matrix, 1);
}

void constant(std::vector<std::vector<float> > *matrix, float value){
    for(int i=0; i<matrix->size(); i++){
        for(int j=0; j<matrix->at(i).size(); j++){
            matrix->at(i).at(j) = value;
        }
    }
}

void glorot_uniform(std::vector<std::vector<float> > *matrix, int input_size, int output_size){
    float limit = std::sqrt(6.0) / std::sqrt(input_size + output_size);
    random_uniform(matrix, -limit, limit);
}