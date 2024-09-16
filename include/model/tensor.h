#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>

struct Tensor{
    public:

        Tensor();
        
        Tensor(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0 );

        void clear();

        std::vector<int> shape();

        int shape(int index);

        double* data();

        int size();

        double& operator[](int index);
        
        void reshape(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0);

    private:
        std::vector<int> m_shape;
        std::vector<double> m_data;
};

#endif // TENSOR_H