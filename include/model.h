#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "data_loader.h"
#include "layers.h"


class Model{
    public:

        Model();
        void add(Layer *layer);

        std::vector<std::vector<float> > forward(std::vector<std::vector<float> > *input);
        void summary();
        void compile();
        void train(DataLoader *x, int epochs, int batch_size, float learning_rate);
        void save(std::string filename);
        void load(std::string filename);
        std::vector<float> mse(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2);
        std::vector<float> loss(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2);

        Layer* operator[](int index);

    private:
        std::vector<Layer*> layers;
        int input_size;

        std::vector<float> get_one_hot(int label, int size);

};

#endif // MODEL_H