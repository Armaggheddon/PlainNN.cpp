#ifndef MODEL_MODEL_H
#define MODEL_MODEL_H

#include "layers.h"
#include "lr_scheduler.h"
#include "data_loaders.h"
#include <vector>
#include <chrono>

/**
 * @brief Struct to hold the result of an evaluation
 */
struct EvaluationResult{
    int correct;        // @brief The number of correct predictions
    int total;          // @brief The total number of predictions
    double accuracy;    // @brief The accuracy of the model
    double avg_loss;    // @brief The average loss of the model

    std::vector<double> avg_loss_per_class; // @brief The average loss per class
};


/**
 * @brief Class that represent a neural network model
 */
class Model{
    public:
        Model();
        ~Model(){};

        /**
         * @brief Set the learning rate scheduler
         * 
         * @param scheduler The learning rate scheduler
         */
        void set_lr_scheduler(LRScheduler* scheduler);

        /**
         * @brief Add a layer to the model
         * 
         * @param layer The layer to add
         * 
         * @note The layers should be added in the order they
         * should be executed in the forward pass. The first layer
         * must be an input layer.
         */
        void add_layer(Layer* layer);

        /**
         * @brief Start training the model
         * 
         * @param train_dataloader The dataloader for the training data
         * @param learning_rate The learning rate
         * @param epochs The number of epochs to train for
         * @param batch_size The size of the batches
         * @param save_checkpoint Whether to save a checkpoint of the model, default is false
         * @param checkpoint_path The path to save the checkpoint to, empty by default
         * 
         * @note This function will train the model for the specified number of epochs
         * using the specified learning rate and batch size. The model will be trained
         * on the data provided by the train_dataloader. If save_checkpoint is true, the
         * model will be saved to the checkpoint_path after each epoch according to the
         * format `checkpoint_path_epoch_i.weights` and `checkpoint_path_epoch_i.json`.
         */
        void train(
            DataLoader& train_dataloader,
            double learning_rate,
            int epochs,
            int batch_size,
            bool save_checkpoint = false,
            std::string checkpoint_path = ""
        );

        /**
         * @brief Start training the model
         * 
         * @param train_dataloader The dataloader for the training data
         * @param test_dataloader The dataloader for the test data
         * @param learning_rate The learning rate
         * @param epochs The number of epochs to train for
         * @param batch_size The size of the batches
         * @param save_checkpoint Whether to save a checkpoint of the model, default is false
         * @param checkpoint_path The path to save the checkpoint to, empty by default
         * 
         * @note This function will train the model for the specified number of epochs
         * using the specified learning rate and batch size. After each epoch a validation
         * run is performed using the `test_dataloader`. The model will be trained
         * on the data provided by the train_dataloader. If save_checkpoint is true, the
         * model will be saved to the checkpoint_path after each epoch according to the
         * format `checkpoint_path_epoch_i.weights` and `checkpoint_path_epoch_i.json`.
         */
        void train(
            DataLoader& train_dataloader,
            DataLoader& test_dataloader,
            double learning_rate,
            int epochs,
            int batch_size,
            bool save_checkpoint = false,
            std::string checkpoint_path = ""
        );

        /**
         * @brief Evaluate the model on the data provided by the dataloader
         * 
         * @param dataloader The dataloader to get the data from
         * @param show_output Whether to show the output of the model
         * @param indent Whether to indent the output, false by default. (Callers should not set this)
         * 
         * @return EvaluationResult The result of the evaluation
         */
        EvaluationResult evaluate(DataLoader& dataloader, bool show_output = true, bool indent = false);

        /**
         * @brief Forward pass of the model
         * 
         * @param input The input to the model
         * @return Tensor The output of the model
         */
        Tensor forward(Tensor& input);

        /**
         * @brief Prints a summary of the model to the console
         * in a table formatted as follows:
         * - Layer Name
         * - Layer Type
         * - Output Shape
         * - Number of Parameters
         * 
         * Also prints the total number of parameters and their size
         */
        void summary();

        /**
         * @brief Get a layer from the model
         * 
         * @param index The index of the layer
         * 
         * @return Layer* The layer at the specified index
         */
        Layer* get_layer(int index);

        /**
         * @brief Saves the model to disk. The model is saved in two parts:
         * - The architecture of the model is saved in a JSON file with the extension `.json`
         * - The weights of the model are saved in a binary file with the extension `.weights`
         * 
         * @param file_name The name of the file to save the model to, without the extension
         * @param weights_only Whether to save only the weights of the model, default is false
         * 
         * @note If weights_only is true, only the weights of the model are saved.
         */
        void save(std::string file_name, bool weights_only = false);

        /**
         * @brief Load the model from disk. The model is loaded in two parts:
         * - The architecture of the model is loaded from a JSON file with the extension `.json`
         * - The weights of the model are loaded from a binary file with the extension `.weights`
         * 
         * @param file_name The name of the file to load the model from, without the extension
         * @param weights_only Whether to load only the weights of the model, default is false
         * 
         * @note If weights_only is true, only the weights of the model are loaded.
         */
        void load(std::string file_name, bool weights_only = false);

    private:
        std::vector<Layer*> m_layers;

        void _train(
            DataLoader* train_dataloader,
            DataLoader* test_dataloader,
            double learning_rate,
            int epochs,
            int batch_size,
            bool save_checkpoint,
            std::string checkpoint_path
        );

        LRScheduler *m_lr_scheduler;

        /**
         * @brief Converts a count to a size in a human readable format
         * 
         * @param count The count to convert
         * @param buff The buffer to write the result to
         * @param buff_size The size of the buffer
         * @param size The size to convert to, default is 0
         * 
         * @note If size is not provided, it is assumed that count already represents a size,
         * and the function will simply convert it to a human readable format. If size is provided,
         * the function will convert count to a size according to `count*size` and then convert
         * it to a human readable format.
         */
        void count_to_size(int count, char* buff, size_t buff_size, size_t size = 0);

        /**
         * @brief Prints a progress bar to the console formatted as follows:
         * {curr_progress}/{max_progress} [{bar}] {trailing_message}
         * 
         * @param curr_progress The current progress
         * @param max_progress The maximum progress
         * @param trailing_message The message to print after the progress bar
         * @param width The width of the progress bar
         * @param indent Whether to indent the progress bar
         */
        void print_progress(int curr_progress, int max_progress, std::string trailing_message = "", int width = 50, bool indent = true);

        /**
         * @brief Converts a duration to a human readable format
         * 
         * @param duration The duration to convert
         * @param buff The buffer to write the result to
         */
        void make_duration_readable(const std::chrono::duration<double>& duration, char* buff, size_t buff_size);
};

#endif // MODEL_MODEL_H