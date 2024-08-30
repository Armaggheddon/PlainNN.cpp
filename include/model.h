#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "data_loader.h"
#include "layers.h"


class Model{
    public:
        int layers_count;
        Model();
        void add(Layer *layer);

        std::vector<std::vector<float> > forward(std::vector<std::vector<float> > *input);
        void summary();
        void compile();
        void train(DataLoader *x, int epochs, int batch_size, float learning_rate, std::string checkpoints_path = nullptr);
        void save(std::string filename);
        void load(std::string filename);
        std::vector<float> mse(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2);

        Layer* operator[](int index);

    private:
        std::vector<Layer*> layers;
        int input_size;

        std::vector<float> _get_one_hot(int label, int size);
        std::vector<LayerSummary> _parse_json(std::string filename);
        void _parse_weights(std::string filename);
        void _build_model_from_layer_summary(std::vector<LayerSummary> layer_summary);

};

#endif // MODEL_H