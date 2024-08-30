#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include "model.h"

class ModelLoader{
    public:
        static Model loadJson(const std::string& path);
        static void loadWeights(const std::string& path, Model &model);
};

#endif // MODEL_LOADER_H