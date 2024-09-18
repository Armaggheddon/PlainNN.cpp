#ifndef MODEL_STORAGE_H
#define MODEL_STORAGE_H

#include "model/layers/layers.h"
#include "model/model.h"
#include <string>
#include <vector>


const std::string MODEL_ARCH_FILE_EXT = ".json";
const std::string MODEL_WEIGHTS_FILE_EXT = ".weights";


class ModelStorage{

    public:
        static void save_model_arch(
            std::string file_name,
            std::vector<LayerSummary> layer_summaries
        );

        static void save_model_weights(
            std::string file_name,
            std::vector<std::vector<double> > weights
        );

        static void load_model_arch(
            std::string file_name,
            Model& model
        );

        static void load_model_weights(
            std::string file_name,
            int layer_count,
            Model& model
        );

};

#endif // MODEL_STORAGE_H