#ifndef MODEL_LAYERS_INITIALIZATION_H
#define MODEL_LAYERS_INITIALIZATION_H

#include <ctime>
#include <random>
#include <cmath>
#include <vector>

class GolorotInitialization{
    public:

        static void initialize(
            std::vector<double>& tensor,
            double dim_sum
        );
};

#endif // MODEL_LAYERS_INITIALIZATION_H