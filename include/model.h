#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layers.h"

class Model{
    public:

        Model();
        void add(Layer *layer);

        std::vector<std::vector<float> > forward(std::vector<std::vector<float> > *input);
        void backward();
        void summary();
        void compile();

        Layer* operator[](int index);

    private:
        std::vector<Layer*> layers;
        int input_size;
};

#endif // MODEL_H