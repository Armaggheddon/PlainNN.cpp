#include "tensor.h"

#include "initialization.h"

#include <initializer_list>
#include <vector>

Tensor::Tensor(){}

Tensor::Tensor(std::initializer_list<int> dims, bool random_init, double fill_value ){
    int data_size = 1;
    int dim_sum = 0;
    for(int dim : dims){
        m_shape.push_back(dim);
        data_size *= dim;
        dim_sum += dim;
    }

    m_data.resize(data_size, fill_value);

    if(random_init) GolorotInitialization::initialize(m_data, dim_sum);
}

void Tensor::clear(){
    std::fill(m_data.begin(), m_data.end(), 0);
}

std::vector<int> Tensor::shape(){
    return m_shape;
}

int Tensor::shape(int index){
    return m_shape[index];
}

double* Tensor::data(){
    return m_data.data();
}

int Tensor::size(){
    return m_data.size();
}

double& Tensor::operator[](int index){
    return m_data[index];
}

void Tensor::reshape(std::initializer_list<int> dims, bool random_init, double fill_value){
    int data_size = 1;
    int dim_sum = 0;

    m_shape.clear();
    m_data.clear();

    for(int dim : dims){
        m_shape.push_back(dim);
        data_size *= dim;
        dim_sum += dim;
    }

    m_data.resize(data_size, 0);

    if(random_init) GolorotInitialization::initialize(m_data, dim_sum);
}