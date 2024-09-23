#include <initializer_list>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "tensor.hpp"
#include "initialization.hpp"

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


Tensor::Tensor(std::vector<int> dims, bool random_init, double fill_value ){
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


Tensor::Tensor(std::initializer_list<int> dims, std::vector<double>& data){
    int data_size = 1;
    for(int dim : dims){
        m_shape.push_back(dim);
        data_size *= dim;
    }
    m_data = data;
}

Tensor::Tensor(std::vector<int> dims, std::vector<double>& data){
    m_shape = dims;
    m_data = data;
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

    m_data.resize(data_size, fill_value);

    if(random_init) GolorotInitialization::initialize(m_data, dim_sum);
}


std::string Tensor::shape_str(){
    std::string str;
    str += "(";

    for(size_t i = 0; i < m_shape.size(); i++){
        str += std::to_string(m_shape[i]);
        if(i != m_shape.size() - 1) str += ", ";
    }

    str += ")";
    return str;
}