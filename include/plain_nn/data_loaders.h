#ifndef PLAIN_NN_DATA_LOADERS_H
#define PLAIN_NN_DATA_LOADERS_H

#include "tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <random>

/**
 * @brief Struct to hold a single item in a dataset
 */
struct DatasetItem{
    Tensor data;
    int target;
};

/**
 * @brief Struct to hold a batch of data
 */
struct BatchData{
    std::vector<Tensor> input_data;
    std::vector<Tensor> targets_one_hot;
    std::vector<int> targets_idx;
};

/**
 * @brief One hot encode a label
 * 
 * @param label_idx The index of the label
 * @param num_classes The number of classes
 * @return Tensor The one hot encoded label
 * 
 * @note This function is useful for single label classification tasks.
 * For multi-label classification tasks, implement a different encoding scheme.
 */
Tensor one_hot_encode(int label_idx, int num_classes);

/**
 * @brief Abstract class for data loaders. New data loaders
 * should inherit from this class and implement all of its methods.
 */
class DataLoader{
    public:

        ~DataLoader(){};

        /**
         * @brief Load the data into memory
         */
        virtual void load() = 0;

        /**
         * @brief Get a batch of data
         * 
         * @param batch_size The size of the batch
         * @return BatchData The batch of data
         * 
         */
        virtual BatchData get_batch(int batch_size) = 0;
        
        /**
         * @brief Start a new epoch. The data loader should
         * shuffle the data if necessary and reset the offset
         * to the beginning of the dataset.
         */
        virtual void new_epoch() = 0;

        /**
         * @brief Get the number of classes in the dataset
         * 
         * @return int The number of classes
         */
        virtual int num_classes() = 0;

        /**
         * @brief Shuffle the dataset
         */
        virtual void shuffle() = 0;

        /**
         * @brief Get the number of steps per epoch
         * 
         * @param batch_size The size of the batch
         * @return int The number of steps required to complete
         * a single epoch
         */
        virtual int steps_per_epoch(int batch_size) = 0;
};

/**
 * @brief MNIST data loader, can also be used for Fashion MNIST
 */
class MNISTDataLoader : public DataLoader{
    public:

        /**
         * @brief Construct a new MNISTDataLoader object
         * 
         * @param data_path The path to the images file
         * @param labels_path The path to the labels file
         * @param shuffle Whether to shuffle the dataset
         * @param drop_last Whether to drop the last batch if it is smaller than the batch size
         */
        MNISTDataLoader(
            std::string data_path, 
            std::string labels_path,
            bool shuffle = true,
            bool drop_last = true);
        
        BatchData get_batch(int batch_size);
        void new_epoch();
        void load();
        int num_classes();
        void shuffle();

        int steps_per_epoch(int batch_size);
    private:
        std::vector<DatasetItem> m_dataset;
        std::string m_data_path;
        std::string m_labels_path;

        int m_offset;
        bool m_shuffle, m_drop_last;

        std::default_random_engine rng;

        /**
         * @brief Load the images data into memory
         */
        void load_data();

        /**
         * @brief Load the labels data into memory
         */
        void load_labels();
};

#endif // PLAIN_NN_DATA_LOADERS_H