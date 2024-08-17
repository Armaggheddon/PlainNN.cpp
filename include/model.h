#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "data_loader.h"
#include "layers.h"


class Model{
    public:

        Model();
        void add(Layer *layer);

        std::vector<std::vector<double> > forward(std::vector<std::vector<double> > *input);
        void summary();
        void compile();
        void train(DataLoader *x, int epochs, int batch_size, double learning_rate);
        void save(std::string filename);
        void load(std::string filename);

        Layer* operator[](int index);

    private:
        std::vector<Layer*> layers;
        int input_size;

        std::vector<double> get_one_hot(int label, int size);

};

#endif // MODEL_H