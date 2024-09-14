#ifndef DATA_LOADERS_H
#define DATA_LOADERS_H   

#include <vector>
#include <string>
#include <random>
#include <algorithm>


template <class C>
struct DatasetItem{
    std::vector<C> data;
    int target;
};

template <class C>
struct BatchData{
    std::vector<C> input_data;
    std::vector<std::vector<double> > targets_one_hot;
    std::vector<int> targets_idx;
};


std::vector<double> one_hot_encode(int label_idx, int num_classes);


template <class C>
class DataLoader{
    public:
        virtual void load() = 0;
        virtual BatchData<C> get_batch(int batch_size) = 0;
        virtual void new_epoch() = 0;

        virtual int num_classes() = 0;

        virtual void shuffle() = 0;

        virtual int steps_per_epoch(int batch_size) = 0;
};

class MNISTDataLoader : public DataLoader<std::vector<double> >{
    public:
        MNISTDataLoader(
            std::string data_path, 
            std::string labels_path,
            bool shuffle = true,
            bool drop_last = true);
        
        BatchData<std::vector<double> > get_batch(int batch_size);
        void new_epoch();
        void load();
        int num_classes();
        void shuffle();

        int steps_per_epoch(int batch_size);
    private:
        std::vector<DatasetItem<double> > m_dataset;
        std::string m_data_path;
        std::string m_labels_path;
        bool m_shuffle;
        bool m_drop_last;
        int m_offset;

        std::default_random_engine rng;

        void load_data();
        void load_labels();
};



#endif // DATA_LOADERS_H