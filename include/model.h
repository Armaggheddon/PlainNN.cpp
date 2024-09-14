#ifndef MODEL_H
#define MODEL_H

#include "layers.h"
#include "data_loaders.h"
#include <vector>
#include <chrono>
#include <fstream>

struct EvaluationResult{
    int correct;
    int total;
    double accuracy;
    double avg_loss;

    std::vector<double> loss_per_class;
};

class Model{

    public:
        std::vector<Layer*> layers;
        Model(){};

        void save_weights(std::string weights_path){
            _save_weights(weights_path);
        }

        void load_weights(std::string weights_path){
            _load_weights(weights_path);
        }

        Layer* const operator[](int idx){
            return layers[idx];
        }

        void add_layer(Layer* layer){
            if(layers.size() == 0 && layer->config.type != LayerType::INPUT){
                std::printf("The first layer must be an input layer\n");
                return;
            }

            layers.push_back(layer);

            if(!layers.back()->is_initialized){
                // Fall back to casting layer to supported type
                // inizialization depends on the layer type
                if(layer->config.type == LayerType::DENSE){
                    Dense* dense_layer = dynamic_cast<Dense*>(layers.back());
                    if(dense_layer == nullptr){
                        std::printf("Error during cast from Layer to Dense\n");
                        return;
                    }
                    dense_layer->initialize(layers[layers.size()-2]->config.output_size);
                }
            }
        }

        void summary(){
            std::printf("_________________________________________________________________\n");
            std::printf("Layer (type)                 Output Shape              Param #\n");
            std::printf("=================================================================\n");

            int total_params = 0;

            for(int i = 0; i < layers.size(); i++){
                Layer* layer = layers[i];
                std::string layer_name = "[" + std::to_string(i) + "] " + layer->name();
                std::string output_shape = "( 1, " + std::to_string(layer->config.output_size) + ")";
                int num_params = 0;

                if(layer->config.type == LayerType::DENSE){
                    Dense* dense_layer = dynamic_cast<Dense*>(layer);
                    if(dense_layer == nullptr){
                        std::printf("Error during cast from Layer to Dense\n");
                        return;
                    }
                    num_params = dense_layer->config.input_size * dense_layer->config.output_size + dense_layer->config.output_size;
                }

                total_params += num_params;

                std::printf("%-24s %-24s %12d\n", layer_name.c_str(), output_shape.c_str(), num_params);
            }

            std::printf("=================================================================\n");

            char trainable_params_buff[16], non_trainable_params_buff[16], total_params_buff[16];
            num_params_to_size(total_params * sizeof(double), total_params_buff);
            num_params_to_size(total_params * sizeof(double), trainable_params_buff);
            num_params_to_size(0 * sizeof(double), non_trainable_params_buff);

            std::printf("Total params: %d (%s)\n", total_params, total_params_buff);
            std::printf("Trainable params: %d (%s)\n", total_params, trainable_params_buff);
            std::printf("Non-trainable params: 0 (%s)\n", non_trainable_params_buff);
            std::printf("_________________________________________________________________\n");
        }

        template <class C>
        std::vector<double> forward(C& input){
            std::vector<double> output = input;
            for(int i = 1; i < layers.size(); i++){
                output = layers[i]->forward(&output);
            }
            return output;
        }
        
        template <class C>
        std::vector<std::vector<double> > forward_batch(std::vector<C>& input){
            std::vector<std::vector<double> > output;
            for(int i = 0; i < input.size(); i++){
                output.push_back(forward(input[i]));
            }
            return output;
        }

        template <class C>
        EvaluationResult evaluate(DataLoader<C>& dataloader, bool show_output = true){
            return _evaluate(&dataloader, show_output, false);
        }

        template <class C>
        void train(DataLoader<C>& train_dataloader, DataLoader<C>& test_dataloader, double learning_rate, int epochs, int batch_size){
            _train(&train_dataloader, &test_dataloader, learning_rate, epochs, batch_size);
        }

        template <class C>
        void train(DataLoader<C> & dataloader, double learning_rate, int epochs, int batch_size){
            _train(&dataloader, nullptr, learning_rate, epochs, batch_size);
        };

    private:
        
        template <class C>
        void _train(DataLoader<C>* train_dataloader, DataLoader<C>* test_dataloader, double learning_rate, int epochs, int batch_size){
            int steps_per_epoch = train_dataloader->steps_per_epoch(batch_size);
            char trailing_message_buff[128];
            char running_time_buff[16], step_time_buff[16];

            for(int epoch = 0; epoch < epochs; epoch++){
                
                auto epoch_s_time = std::chrono::system_clock::now();
                std::printf("Epoch %d/%d\n", epoch+1, epochs);

                for(int step = 0; step < steps_per_epoch; step++){
                    
                    auto step_s_time = std::chrono::system_clock::now();

                    BatchData<C> batch = train_dataloader->get_batch(batch_size);
                    if(batch.input_data.size() == 0){
                        // If the batch is empty, it means that the train_dataloader has reached the end of the dataset
                        continue;
                    }
                    std::vector<C> batch_inputs = batch.input_data; 
                    std::vector<std::vector<double> > batch_targets = batch.targets_one_hot;
                    std::vector<int> batch_targets_idx = batch.targets_idx;

                    double error = 0;
                    int correct = 0;

                    for(int b = 0; b < batch_inputs.size(); b++){
                        std::vector<double> output = forward(batch_inputs[b]);

                        for(int i = 0; i < output.size(); i++){
                            error += 0.5 * std::pow(output[i] - batch_targets[b][i], 2);
                        }

                        int max_idx = std::max_element(output.begin(), output.end()) - output.begin();
                        if(max_idx == batch_targets_idx[b]){
                            correct++;
                        }

                        int last_layer_idx = layers.size() - 1;
                        std::vector<double> next_layer_grads;
                        for(int layer = last_layer_idx; layer > 0; layer--){
                            next_layer_grads = this->layers[layer]->backward(
                                layer == 1 ? &batch_inputs[b] : &layers[layer-1]->output,
                                layer == last_layer_idx ? nullptr : &layers[layer + 1]->params.weights,
                                layer == last_layer_idx ? &batch_targets[b] : &next_layer_grads
                            )[0]; // TODO change function signature to return a single vector
                        }
                    }

                    for(int layer = 1; layer < layers.size(); layer++){
                        layers[layer]->step(learning_rate, batch_size);
                    }

                    auto step_e_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> step_duration = step_e_time - step_s_time;
                    std::chrono::duration<double> running_epoch_time = step_e_time - epoch_s_time;
                    
                    make_duration_readable(step_duration, step_time_buff);
                    make_duration_readable(running_epoch_time, running_time_buff);

                    std::sprintf(trailing_message_buff, "%s %s/step - Error: %.04f - Accuracy: %.04f", 
                        running_time_buff, step_time_buff, error/batch_size, (double)correct/batch_size);
                    print_progress(step, steps_per_epoch, trailing_message_buff, 20);
                }
                std::printf("\n");

                train_dataloader->new_epoch();

                if(test_dataloader != nullptr){
                    _evaluate(test_dataloader, true, true);
                }
            }
        }

        template <class C>
        EvaluationResult _evaluate(DataLoader<C>* dataloader, bool show_output, bool indent){
            
            int correct = 0, total = 0;
            int total_steps = dataloader->steps_per_epoch(1);
            double accuracy = 0, loss = 0, tmp_loss = 0;

            auto running_s_time = std::chrono::system_clock::now();
            auto step_s_time = std::chrono::system_clock::now();

            char step_time_buff[16], running_time_buff[16];

            std::vector<double> loss_per_class(dataloader->num_classes(), 0);
            
            char message_buff[128];
            for(int step = 0; step < total_steps; step++){

                step_s_time = std::chrono::system_clock::now();

                BatchData<C> batch = dataloader->get_batch(1);
                if(batch.input_data.size() == 0){
                    // If the batch is empty, it means that the dataloader has reached the end of the dataset
                    continue;
                }

                // we are sampling only one element at a time
                C input = batch.input_data[0];
                int target = batch.targets_idx[0];

                std::vector<double> output = forward(input);
                int max_idx = std::max_element(output.begin(), output.end()) - output.begin();

                if(max_idx == target){
                    correct++;
                }

                for(int i = 0; i < output.size(); i++){
                    tmp_loss = 0.5 * std::pow(output[i] - (i == target ? 1 : 0), 2);
                    loss += tmp_loss;
                    loss_per_class[i] += tmp_loss;
                }

                if(show_output){

                    auto step_e_time = std::chrono::system_clock::now();

                    std::chrono::duration<double> step_duration = step_e_time - step_s_time;
                    std::chrono::duration<double> running_duration = step_e_time - running_s_time;

                    make_duration_readable(step_duration, step_time_buff);
                    make_duration_readable(running_duration, running_time_buff);

                    std::sprintf(message_buff, "Loss: %.04f - Accuracy: %.04f - %s elapsed - %s/step", 
                        loss/step, (double)correct/step, running_time_buff, step_time_buff);
                    print_progress(step+1, total_steps, message_buff, 20, indent);
                }
            }
            if(show_output) {
                printf("\n");
            }

            accuracy = (double)correct / total_steps;

            std::for_each(loss_per_class.begin(), loss_per_class.end(), [total_steps](double& loss){
                loss /= total_steps;
            });

            // Reset the dataloader to the beginning of the dataset
            dataloader->new_epoch();

            return EvaluationResult{
                correct, total_steps, accuracy,
                loss/total_steps,
                loss_per_class};
        }

        void num_params_to_size(int num_params, char* buff){
            if(num_params >= 1000000){
                std::sprintf(buff, "%.02fMB", num_params/1000000.0);
            }else if(num_params >= 1000){
                std::sprintf(buff, "%.02fKB", num_params/1000.0);
            }else{
                std::sprintf(buff, "%dB", num_params);
            }
        }

        void print_progress(int curr_progress, int total, std::string trailing_message = "", int width = 50, bool indent = true){
            char progress_buff[50];
            int progress = (int)((curr_progress / (double)total) * width);
            for(int i = 0; i < width; i++){
                progress_buff[i] = i < progress ? '=' : ' ';
            }
            progress_buff[width] = '\0';
            std::printf("\r%s%d/%d [%s] %s", (indent) ? "    " : "", curr_progress, total, progress_buff, trailing_message.c_str());
            std::fflush(stdout);
        }

        void make_duration_readable(const std::chrono::duration<double>& duration, char* buff){
            
            long int us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            if(us >= 1000000){
                // If step duration is greater than 1 second,
                // print the time formatted in seconds
                std::sprintf(buff, "%4lds", us/1000000);
            }else if (us >= 1000){
                // If step duration is greater than 1 millisecond,
                // print the time formatted in milliseconds
                std::sprintf(buff, "%3ldms", us/1000);
            }else{
                // If step duration is less than 1 millisecond,
                // print the time formatted in microseconds
                std::sprintf(buff, "%3ldus", us);
            }
        }

        void _save(std::string save_path){
            if(!_is_path_valid(save_path, false)){
                std::printf("Invalid path: %s\n", save_path.c_str());
                return;
            }

            _save_weights(save_path + ".weights");
            _save_architecture(save_path + ".json");
        }

        void _save_weights(std::string weights_path){
            if(_is_dir(weights_path)){
                std::printf("Invalid path: %s\n", weights_path.c_str());
                return;
            }

            if(!_has_extension(weights_path, true)){
                weights_path += ".weights";
            }

            std::ofstream file(weights_path, std::ios::binary);
            if(!file.is_open()){
                std::printf("Failed to open file: %s\n", weights_path.c_str());
                return;
            }

            for(Layer* layer : layers){
                std::vector<double> layer_params = layer->get_saveable_params();
                file.write((char*)layer_params.data(), layer_params.size() * sizeof(double));
            }

        }
        
        void _save_architecture(std::string architecture_path);
        // The path with the model name without the extension
        void _load(std::string load_path);
        
        void _load_weights(std::string weights_path){
            if(_is_dir(weights_path)){
                std::printf("Invalid path: %s\n", weights_path.c_str());
                return;
            }

            if(!_has_extension(weights_path, true)){
                weights_path += ".weights";
            }

            std::ifstream file(weights_path, std::ios::binary);
            if(!file.is_open()){
                std::printf("Failed to open file: %s\n", weights_path.c_str());
                return;
            }

            for(int i=1; i < layers.size(); i++){
                Layer* layer = layers[i];
                int buff_size = layer->config.input_size * layer->config.output_size + layer->config.output_size;
                std::vector<double> layer_params(buff_size);
                file.read((char*)layer_params.data(), layer_params.size() * sizeof(double));
                layer->load_weights_and_biases_from_vector(layer_params);
            }
        }

        void _load_architecture(std::string architecture_path);
        
        bool _is_path_valid(std::string path, bool is_weights){
            if(_is_dir(path) || !_has_extension(path, is_weights)){
                return false;
            }

            return true;
        }

        bool _has_extension(std::string path, bool is_weights){
            std::string extension = is_weights ? ".weights" : ".json";
            
            int pos = path.find_last_of('.');
            if(pos == std::string::npos){
                return false;
            }

            std::string path_extension = path.substr(pos);
            return path_extension == extension;
        }

        bool _is_dir(std::string path){
            // Windows ans Posix like systems handle directory differently;
            // Windows uses backslashes and Posix uses forward slashes

            char last_char = path[path.size() - 1];
            return last_char == '/' || last_char == '\\';
        }
};

#endif // MODEL_H