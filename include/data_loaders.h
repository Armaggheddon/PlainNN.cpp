#ifndef DATA_LOADERS_H
#define DATA_LOADERS_H

#include "tensor.h"
#include <vector>
#include <memory>


struct DatasetItem{
    Tensor data;
    int target;
};

struct BatchData{
    std::vector<Tensor> input_data;
    std::vector<Tensor> targets_one_hot;
    std::vector<int> targets_idx;
};

Tensor one_hot_encode(int label_idx, int num_classes);

class DataLoader{
    public:
        virtual void load() = 0;
        virtual BatchData get_batch(int batch_size) = 0;
        virtual void new_epoch() = 0;
        virtual int num_classes() = 0;
        virtual void shuffle() = 0;
        virtual int steps_per_epoch(int batch_size) = 0;
};

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

#endif // DATA_LOADERS_H