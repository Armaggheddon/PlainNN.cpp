#include "initialization.h"
#include <cmath>
#include <vector>

void GolorotInitialization::initialize(
    std::vector<double>& tensor,
    double dim_sum
){
    std::srand(time(NULL));
    double limit_ih = std::sqrt(6.0/dim_sum);

    for(size_t i=0; i<tensor.size(); i++){
        tensor[i] = ((double) std::rand() / RAND_MAX) * 2 * limit_ih - limit_ih;
    }
}
