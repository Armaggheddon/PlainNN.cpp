#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <ctime>
#include <random>
#include <cmath>
#include <vector>

class GolorotInitialization{
    public:

        static void initialize(
            std::vector<std::vector<double> >& weights,
            int input_size, int output_size
        ){
            std::srand(time(NULL));
            double limit_ih = std::sqrt(6.0/(input_size + output_size));

            for(int in=0; in<input_size; in++){
                for(int out=0; out<output_size; out++){
                    weights[in][out] = ((double) std::rand() / RAND_MAX) * 2 * limit_ih - limit_ih;
                }
            }
        }
};

#endif // INITIALIZATION_H