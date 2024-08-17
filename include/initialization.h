#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <vector>

void random_uniform(std::vector<std::vector<double> > *matrix, double min, double max);
void random_normal(std::vector<std::vector<double> > *matrix, double mean, double std);
void zeros(std::vector<std::vector<double> > *matrix);
void ones(std::vector<std::vector<double> > *matrix);
void constant(std::vector<std::vector<double> > *matrix, double value);
void glorot_uniform(std::vector<std::vector<double> > *matrix, int input_size, int output_size);


#endif // INITIALIZATION_H