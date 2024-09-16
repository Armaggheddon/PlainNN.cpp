#ifndef DATA_LOADER_H
#define DATA_LOADER_H

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

#endif // DATA_LOADER_H