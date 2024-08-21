#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <random>
#include "data_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MNISTDataLoader::MNISTDataLoader(std::string dataset_root_folder, float split){
    this->train_data = std::vector<DataInfo>();
    this->test_data = std::vector<DataInfo>();
    this->val_data = std::vector<DataInfo>();
    this->train_indexes = std::vector<int>();
    this->val_indexes = std::vector<int>();
    this->dataset_root_folder = dataset_root_folder;
    this->split = split;
}

void MNISTDataLoader::load(){
    std::string train_folder = this->dataset_root_folder + "/train";
    std::string test_folder = this->dataset_root_folder + "/test";

    std::string train_label_file = train_folder + "/train.txt";
    std::string test_label_file = test_folder + "/test.txt";

    FILE *train_file = fopen(train_label_file.c_str(), "r");

    if(train_file == NULL){
        std::printf("Error opening file %s\n", train_label_file.c_str());
        return;
    }

    char image_path[100];
    int label;
    int count = 0;
    std::vector<DataInfo> tmp_data = std::vector<DataInfo>();
    while(fscanf(train_file, "%s %d", image_path, &label) != EOF){
        DataInfo info;
        info.image_path = train_folder + "/" + image_path;
        info.label = label;
        tmp_data.push_back(info);
    }

    fclose(train_file);
    
    int train_count = (int)(tmp_data.size() * this->split);
    int val_count = tmp_data.size() - train_count;

    for (int i=0; i<tmp_data.size(); i++){
        if(i < train_count){
            this->train_indexes.push_back(this->train_data.size());
            this->train_data.push_back(tmp_data[i]);
        }else{
            this->val_indexes.push_back(this->val_data.size());
            this->val_data.push_back(tmp_data[i]);
        }
    }
    tmp_data.clear();

    std::printf("Loaded %d train samples\n", (int)this->train_data.size());
    std::printf("Loaded %d val samples\n", (int)this->val_data.size());

    FILE *test_file = fopen(test_label_file.c_str(), "r");

    if(test_file == NULL){
        std::printf("Error opening file %s\n", test_label_file.c_str());
        return;
    }

    while(fscanf(test_file, "%s %d", image_path, &label) != EOF){
        DataInfo info;
        info.image_path = test_folder + "/" + image_path;
        info.label = label;
        this->test_data.push_back(info);
    }
    fclose(test_file);

    std::printf("Loaded %d test samples\n", (int)this->test_data.size());

}

void MNISTDataLoader::shuffle(){
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(this->train_indexes), std::end(this->train_indexes), rng);
    std::printf("Top 10 train indexes\n");
    std::shuffle(std::begin(this->val_indexes), std::end(this->val_indexes), rng);
}

std::vector<Data> MNISTDataLoader::get_batch(int batch_size){
    
    std::vector<Data> batch = std::vector<Data>();

    int idx = this->current_offset;
    for(int i=0; i<batch_size; i++){

        DataInfo info = this->train_data[this->train_indexes[idx]];
        
        int width, height, channels;
        unsigned char *image = stbi_load(info.image_path.c_str(), &width, &height, &channels, 1);

        if(image == NULL){
            std::printf("Error loading image %s\n", info.image_path.c_str());
            continue;
        }

        // Normalize image and flatten input image
        std::vector<float> input = std::vector<float>(width*height, 0);
        for(int j=0; j<width*height; j++){
            input[j] = (float)image[j]/255.0;
        }

        Data data;
        data.input = input;
        data.label = info.label;
        batch.push_back(data);

        stbi_image_free(image);
        idx = (idx + 1) % this->train_data.size();
    }

    this->current_offset = idx;
    return batch;
}

Data MNISTDataLoader::get_sample(){
    DataInfo info = this->train_data[this->train_indexes[this->current_offset]];
    
    int width, height, channels;
    unsigned char *image = stbi_load(info.image_path.c_str(), &width, &height, &channels, 1);

    if(image == NULL){
        std::printf("Error loading image %s\n", info.image_path.c_str());
        return Data();
    }

    // Normalize image and flatten input image
    std::vector<float> input = std::vector<float>(width*height, 0);
    for(int j=0; j<width*height; j++){
        input[j] = (float)image[j]/255.0;
    }

    Data data;
    data.input = input;
    data.label = info.label;

    stbi_image_free(image);
    return data;
}