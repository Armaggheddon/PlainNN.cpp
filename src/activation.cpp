#include <cmath>
#include <vector>
#include "activation.h"
#include <string>


ReLU::ReLU(){
    this->name = "relu";
}

void ReLU::forward(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::max(matrix->at(i), 0.0f);
}

void ReLU::backward(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = matrix->at(i) > 0 ? 1 : 0;
}

Sigmoid::Sigmoid(){
    this->name = "sigmoid";
}
void Sigmoid::forward(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = 1.0 / (1.0 + std::exp(-matrix->at(i)));
}

void Sigmoid::backward(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = matrix->at(i) * (1 - matrix->at(i));
}

Softmax::Softmax(){
    this->name = "softmax";
}
void Softmax::forward(std::vector<float> *matrix){
    float sum = 0;
    for(int i=0; i<matrix->size(); i++) sum += std::exp(matrix->at(i));
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::exp(matrix->at(i)) / sum;
}

void Softmax::backward(std::vector<float> *matrix){
    for(int i=0; i<matrix->size(); i++) matrix->at(i) = matrix->at(i) * (1 - matrix->at(i));
}

// void relu(std::vector<float> *matrix){
//     for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::max(matrix->at(i), 0.0f);
// }

// void sigmoid(std::vector<float> *matrix){
//     for(int i=0; i<matrix->size(); i++) matrix->at(i) = 1.0 / (1.0 + std::exp(-matrix->at(i)));
// }

// void softmax(std::vector<float> *matrix){
//     float sum = 0;
//     for(int i=0; i<matrix->size(); i++) sum += std::exp(matrix->at(i));
//     for(int i=0; i<matrix->size(); i++) matrix->at(i) = std::exp(matrix->at(i)) / sum;
// }
