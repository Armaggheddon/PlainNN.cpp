#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <vector>

void random_uniform(std::vector<std::vector<float> > *matrix, float min, float max);
void random_normal(std::vector<std::vector<float> > *matrix, float mean, float std);
void zeros(std::vector<std::vector<float> > *matrix);
void ones(std::vector<std::vector<float> > *matrix);
void constant(std::vector<std::vector<float> > *matrix, float value);
void glorot_uniform(std::vector<std::vector<float> > *matrix, int input_size, int output_size);


#endif // INITIALIZATION_H