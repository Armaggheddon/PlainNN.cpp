#ifndef PLAIN_NN_STORAGE_H
#define PLAIN_NN_STORAGE_H

#include "layers.hpp"
#include "plain_nn.hpp"
#include <string>
#include <vector>


const std::string MODEL_ARCH_FILE_EXT = ".json";
const std::string MODEL_WEIGHTS_FILE_EXT = ".weights";

/**
 * @brief Class to handle the storage of a model
 * object to disk. This class is used to save and load
 * the architecture and weights of a model.
 */
class ModelStorage{

    public:

        /**
         * @brief Save the architecture of a model to disk
         * 
         * @param file_name The name of the file to save the model to, without the extension
         * @param layer_summaries The summaries of the layers in the model
         * 
         * @note The model architecture is saved in JSON format.
         * The final file will have the extension `.json`.
         */
        static void save_model_arch(
            std::string file_name,
            std::vector<LayerSummary> layer_summaries
        );

        /**
         * @brief Save the weights of a model to disk
         * 
         * @param file_name The name of the file to save the model to, without the extension
         * @param weights The weights of the model
         * 
         * @note The model weights are saved in a binary format. The
         * final file will have the extension `.weights`.
         */
        static void save_model_weights(
            std::string file_name,
            std::vector<std::vector<double> > weights
        );

        /**
         * @brief Load the architecture of a model from disk
         * 
         * @param file_name The name of the file to load the model from, without the extension
         * @param model The model to load the architecture into
         * 
         * @note The model architecture is loaded from a JSON file.
         * The `.json` extension is added to the file name.
         */
        static void load_model_arch(
            std::string file_name,
            PlainNN& model
        );

        /**
         * @brief Load the weights of a model from disk
         * 
         * @param file_name The name of the file to load the model from, without the extension
         * @param layer_count The number of layers in the model
         * 
         * @note The model weights are loaded from a binary file.
         * The `.weights` extension is added to the file name.
         */
        static void load_model_weights(
            std::string file_name,
            int layer_count,
            PlainNN& model
        );

};

#endif // PLAIN_NN_STORAGE_H