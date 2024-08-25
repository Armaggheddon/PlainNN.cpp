#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>

typedef struct{
    std::string image_path;
    int label;
} DataInfo;

typedef struct{
    std::vector<float> input;
    int label;
} Data;


class DataLoader{
    public:
        std::vector<DataInfo> train_data;
        std::vector<DataInfo> test_data;
        std::vector<DataInfo> val_data;
        std::string dataset_root_folder;
        bool auto_shuffle = true;
        float split;
        virtual ~DataLoader(){};
        virtual void load() = 0;
        virtual void shuffle() = 0;
        /**
         * @brief Get a batch of data, should return 
         * a empty vector when the offset is greater than the
         * size of the dataset. When called again, it should
         * restart the offset. So that the DataLoader can be
         * used in a loop.
         * 
         * @param batch_size 
         * @return std::vector<Data> 
         */
        virtual std::vector<Data> get_batch(int batch_size) = 0;
        virtual void new_epoch() = 0;
        virtual Data get_sample() = 0;
};

class MNISTDataLoader : public DataLoader{
    public:
        MNISTDataLoader(std::string dataset_root_folder, float split, bool auto_shuffle=false);
        virtual void load();
        virtual void shuffle();
        virtual std::vector<Data> get_batch(int batch_size);
        virtual void new_epoch();
        virtual Data get_sample();
    private:
        int current_offset = 0;
        std::vector<int> train_indexes;
        std::vector<int> val_indexes;
};


#endif // DATALOADER_H