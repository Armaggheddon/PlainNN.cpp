#ifndef INITIALIZATION_H
#define INITIALIZATION_H

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

#endif // INITIALIZATION_H