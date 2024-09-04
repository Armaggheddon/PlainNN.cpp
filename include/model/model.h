#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "data_loader.h"
#include "layers.h"


class Model{
    public:
        int layers_count;

        /**
         * @brief Create a Model object from a JSON file, 
         * only loads the model architecture, not the weights.
         * The weights are initialized randomly.
         * To load the weights as well, refer to the from_chepoint method.
         * 
         * @param filename
         * @return Model
         */
        static Model from_json(const std::string filename);
        
        /**
         * @brief Create a Model object from a JSON file and a weights file.
         * Assumes that the weights file is in the same format as the one saved
         * by the save method. Which means that both the JSON and the weights file
         * have the same name and differ only by the extension, i.e.: model.json and model.weights.
         * 
         * @param json_filename common file name for the JSON and weights files without the extension
         * @return Model
         */
        static Model from_checkpoint(const std::string filename);

        /**
         * @brief Create a Model object from a JSON file and a weights file. It is
         * the same as the from_checkpoint method, but allows the user to specify
         * the name of the weights file in case it is different from the default or 
         * when the weights file has been downloaded from the internet. This
         * methods supports scenarios where the file names are different
         * 
         * @param json_filename json file name with the .json extension
         * @param weights_filename weights file name with the .weights extension
         * @return Model
         */
        static Model from_checkpoint(const std::string json_filename, const std::string weights_filename);

        Model();
        void add(Layer *layer);

        std::vector<std::vector<float> > forward(std::vector<std::vector<float> > *input);
        void summary();
        void initialize();
        void train(DataLoader *x, int epochs, int batch_size, float learning_rate, std::string checkpoints_path = nullptr, int from_epoch = 0);
        void save(std::string filename);        
        std::vector<float> mse(std::vector<std::vector<float> > v1, std::vector<std::vector<float> > v2);

        void load_weights(std::string filename);

        Layer* operator[](int index);

    private:
        std::vector<Layer*> layers;
        int input_size;
        bool is_initialized = false;
        std::vector<float> _get_one_hot(int label, int size);
        void _check_initialized();

};

#endif // MODEL_H