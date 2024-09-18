#ifndef MODEL_TENSOR_H
#define MODEL_TENSOR_H

#include <vector>
#include <initializer_list>
#include <string>

struct Tensor{
    public:

        Tensor();

        Tensor(std::vector<int> dims, bool random_init = false, double fill_value = 0 );
        
        Tensor(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0 );

        void clear();

        std::vector<int> shape();

        int shape(int index);

        double* data();

        int size();

        double& operator[](int index);
        
        void reshape(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0);

        std::string shape_str();

    private:
        std::vector<int> m_shape;
        std::vector<double> m_data;
};

#endif // MODEL_TENSOR_H