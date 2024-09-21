#ifndef PLAIN_NN_LAYERS_INITIALIZATION_H
#define PLAIN_NN_LAYERS_INITIALIZATION_H

#include <ctime>
#include <random>
#include <cmath>
#include <vector>

/**
 * @brief Golorot initialization
 * 
 * Uses the following formula:
 * 
 * limit = sqrt(6 / (dim_sum))
 * val = U[-limit, limit] = U[0, 2*limit] - limit
 * 
 * where U[a, b] is a uniform distribution between a and b
 * 
 */
class GolorotInitialization{
    public:

        /**
         * @brief Initialize a tensor with Golorot initialization
         * 
         * @param tensor The tensor to initialize
         * @param dim_sum The sum of the dimensions of the tensor
         */        
        static void initialize(
            std::vector<double>& tensor,
            double dim_sum
        );
};

#endif // PLAIN_NN_LAYERS_INITIALIZATION_H