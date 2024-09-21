#ifndef PLAIN_NN_TENSOR_H
#define PLAIN_NN_TENSOR_H

#include <vector>
#include <initializer_list>
#include <string>

/**
 * @brief Class to represent a tensor
 */
struct Tensor{
    public:

        Tensor();

        /**
         * @brief Construct a new Tensor object
         * 
         * @param dims The dimensions of the tensor
         * @param random_init If true, the tensor will be initialized using the Goolorot initialization
         * @param fill_value The value to fill the tensor with, ignored if random_init is true
         */
        Tensor(std::vector<int> dims, bool random_init = false, double fill_value = 0 );
        
        /**
         * @brief Construct a new Tensor object
         * 
         * @param dims The dimensions of the tensor in the form of an initializer list, e.g. {2, 3, 4}
         * @param random_init If true, the tensor will be initialized using the Goolorot initialization
         * @param fill_value The value to fill the tensor with, ignored if random_init is true
         */
        Tensor(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0 );

        /**
         * @brief Construct a new Tensor object with the specified data
         * 
         * @param dims The dimensions of the tensor
         * @param data The data to fill the tensor with
         */
        Tensor(std::initializer_list<int> dims, std::vector<double>& data);

        /**
         * @brief Clears the contents of the tensor
         * by setting all values to 0
         */
        void clear();

        /**
         * @brief Get the shape of the tensor as a vector
         * 
         * @return std::vector<int> The shape of the tensor
         */
        std::vector<int> shape();

        /**
         * @brief Get the shape of the tensor at a specific index
         * 
         * @param index The index of the shape to get
         * @return int The shape at the index
         */
        int shape(int index);

        /**
         * @brief Get the data of the tensor
         * 
         * @return double* The data of the tensor
         * 
         * @note It is the caller's responsibility to know how to interpret the data
         */
        double* data();

        /**
         * @brief Get the size of the tensor, i.e. the number of elements
         * 
         * @return int The size of the tensor
         */
        int size();

        /**
         * @brief Get the value at the specified index
         * 
         * @param index The index to get the value from
         * @return double& The value at the index
         * 
         * @note If fast access is required, e.g. in a loop, prefer
         * the data() method and access the data directly
         */
        double& operator[](int index);
        
        /**
         * @brief Reshape the tensor
         * 
         * @param dims The new dimensions of the tensor
         * @param random_init If true, the tensor will be initialized using the Goolorot initialization
         * @param fill_value The value to fill the tensor with, ignored if random_init is true
         * 
         * @note The contents of the tensor will be lost after reshaping
         */
        void reshape(std::initializer_list<int> dims, bool random_init = false, double fill_value = 0);

        /**
         * @brief Get a string representation of the shape
         * 
         * @return std::string The shape as a string
         */
        std::string shape_str();

    private:
        std::vector<int> m_shape;
        std::vector<double> m_data;
};

#endif // PLAIN_NN_TENSOR_H