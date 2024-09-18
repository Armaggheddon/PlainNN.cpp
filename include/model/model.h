#ifndef MODEL_MODEL_H
#define MODEL_MODEL_H

#include "layers.h"
#include "lr_scheduler.h"
#include "data_loaders.h"
#include <vector>
#include <chrono>

struct EvaluationResult{
    int correct;
    int total;
    double accuracy;
    double avg_loss;

    std::vector<double> avg_loss_per_class;
};


class Model{
    public:
        Model();
        ~Model(){};

        void set_lr_scheduler(LRScheduler* scheduler);

        void add_layer(Layer* layer);
        void train(
            DataLoader& train_dataloader,
            double learning_rate,
            int epochs,
            int batch_size,
            bool save_checkpoint = false,
            std::string checkpoint_path = ""
        );

        void train(
            DataLoader& train_dataloader,
            DataLoader& test_dataloader,
            double learning_rate,
            int epochs,
            int batch_size,
            bool save_checkpoint = false,
            std::string checkpoint_path = ""
        );

        EvaluationResult evaluate(DataLoader& dataloader, bool show_output = true, bool indent = false);

        Tensor forward(Tensor& input);

        void summary();

        Layer* get_layer(int index);

        void save(std::string file_name, bool weights_only = false);
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

        void count_to_size(int count, char* buff, size_t buff_size, size_t size = 0);
        void print_progress(int curr_progress, int max_progress, std::string trailing_message = "", int width = 50, bool indent = true);
        void make_duration_readable(const std::chrono::duration<double>& duration, char* buff);
};

#endif // MODEL_MODEL_H