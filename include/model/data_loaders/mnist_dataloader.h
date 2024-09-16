#ifndef MNIST_DATALOADER_H
#define MNIST_DATALOADER_H

#include "data_loader.h"
#include <vector>
#include <random>
#include <string>

class MNISTDataLoader : public DataLoader{
    public:
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

        void load_data();
        void load_labels();
};


#endif // MNIST_DATALOADER_H