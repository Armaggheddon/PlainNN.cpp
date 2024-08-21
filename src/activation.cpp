#include <cmath>
#include <vector>
#include "activation.h"


void relu(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::max(matrix->at(i), 0.0f);
}

void sigmoid(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = 1.0 / (1.0 + std::exp(-matrix->at(i)));
}

void softmax(std::vector<float> *matrix){
    float sum = 0;
    for(int i=0; i<matrix->size(); i++) sum += std::exp(matrix->at(i));
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::exp(matrix->at(i)) / sum;
}
