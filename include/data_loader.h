#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>

typedef struct{
    std::string image_path;
    int label;
} DataInfo;

typedef struct{
    std::vector<double> input;
    int label;
} Data;


class DataLoader{
    public:
        std::vector<DataInfo> train_data;
        std::vector<DataInfo> test_data;
        std::vector<DataInfo> val_data;
        std::string dataset_root_folder;
        double split;
        virtual ~DataLoader(){};
        virtual void load() = 0;
        virtual void shuffle() = 0;
        virtual std::vector<Data> get_batch(int batch_size) = 0;
        virtual Data get_sample() = 0;
};

class MNISTDataLoader : public DataLoader{
    public:
        MNISTDataLoader(std::string dataset_root_folder, double split);
        virtual void load();
        virtual void shuffle();
        virtual std::vector<Data> get_batch(int batch_size);
        virtual Data get_sample();
    private:
        int current_offset = 0;
        std::vector<int> train_indexes;
        std::vector<int> val_indexes;
};


#endif // DATALOADER_H