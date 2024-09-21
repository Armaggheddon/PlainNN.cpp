#include "plain_nn.hpp"

#include "layers.hpp"
#include "data_loaders.hpp"
#include "model_storage.hpp"
#include "utils.hpp"

#include <vector>
#include <chrono>
#include <algorithm>
#include <map>

PlainNN::PlainNN(){}

void PlainNN::add_layer(Layer* layer){
    if(m_layers.size() == 0 && layer->layer_type != LayerType::INPUT){
        std::printf("First layer must be an input layer\n");
        exit(1);
    }

    m_layers.push_back(layer);

    if(!m_layers.back()->is_initialized){
        if(m_layers.back()->layer_type == LayerType::DENSE){
            Dense* dense_layer = dynamic_cast<Dense*>(m_layers.back());
            
            // TODO: assert that the output shape of the previous layer
            // is the same as the input shape of the current layer

            dense_layer->initialize({m_layers[m_layers.size()-2]->output.shape().back()});
        }
    }
}


Layer* PlainNN::get_layer(int index){
    return m_layers[index];
}


void PlainNN::freeze_layer(int index, bool freeze){
    m_layers[index]->is_frozen = freeze;
}


void PlainNN::set_lr_scheduler(LRScheduler* scheduler){
    m_lr_scheduler = scheduler;
}


void PlainNN::summary(){
    std::printf("___________________________________________________________\n");
    std::printf("%-12s %-12s %-15s %15s\n", "Layer", "(Type)", "Output Shape", "Param #");
    std::printf("===========================================================\n");

    int total_param_count = 0;

    std::map<int, int> encountered_layers;

    for(size_t i = 0; i < m_layers.size(); i++){
        Layer* layer = m_layers[i];

        LayerSummary summary = layer->get_summary();

        std::string layer_name;

        if(encountered_layers.count(summary.layer_type) == 0){
            encountered_layers[summary.layer_type]++;
            layer_name += string_to_lower(summary.layer_name);
        } else{
            encountered_layers[summary.layer_type]++;
            layer_name += string_to_lower(summary.layer_name) + "_" + std::to_string(encountered_layers[summary.layer_type]-1);
        }

        std::string layer_type = "(" + layer->name() + ")";
        std::string output_shape = layer->output.shape_str();

        int num_params = 0;

        num_params = summary.param_count;

        if(summary.layer_name.compare(LAYER_TYPE_NAMES[LayerType::INPUT]) != 0){
            total_param_count += num_params;
        }

        std::printf("%-12s %-12s %-15s %15d\n", layer_name.c_str(), layer_type.c_str(), output_shape.c_str(), num_params);
    }

    std::printf("===========================================================\n");

    int params_buff_size = 32;
    char trainable_params_buff[params_buff_size], non_trainable_params_buff[params_buff_size], total_params_buff[params_buff_size];
    count_to_size(total_param_count, total_params_buff, params_buff_size, sizeof(double));
    count_to_size(total_param_count, trainable_params_buff, params_buff_size, sizeof(double));
    count_to_size(0, non_trainable_params_buff, params_buff_size, sizeof(double));

    std::printf("Total params: %s\n", total_params_buff);
    std::printf("Trainable params: %s\n", trainable_params_buff);
    std::printf("Non-trainable params: %s\n", non_trainable_params_buff);
    std::printf("___________________________________________________________\n");
}


void PlainNN::save(std::string file_name, bool weights_only){
    if (!weights_only)
    {
        std::vector<LayerSummary> layer_summaries;
        for(size_t i = 0; i < m_layers.size(); i++){
            layer_summaries.push_back(m_layers[i]->get_summary());
        }
        ModelStorage::save_model_arch(file_name, layer_summaries);
    }

    std::vector<std::vector<double>> weights;

    for(size_t i = 0; i < m_layers.size(); i++){
        weights.push_back(m_layers[i]->get_saveable_params());
    }

    ModelStorage::save_model_weights(file_name, weights);
}


void PlainNN::load(std::string file_name, bool weights_only){
    if(!weights_only){

        ModelStorage::load_model_arch(file_name, *this);


    }

    ModelStorage::load_model_weights(file_name, m_layers.size(), *this);
}


Tensor PlainNN::forward(Tensor& input){
    Tensor output = input;

    for(size_t i = 1; i < m_layers.size(); i++){
        output = m_layers[i]->forward(output);
    }

    return output;
}


EvaluationResult PlainNN::evaluate(DataLoader& dataloader, bool show_output, bool indent){
    int correct = 0;
    int total_steps = dataloader.steps_per_epoch(1);
    double accuracy = 0, loss = 0, tmp_loss = 0;

    auto running_s_time = std::chrono::system_clock::now();
    auto step_s_time = std::chrono::system_clock::now();

    size_t time_buff_size = 24;
    char step_time_buff[time_buff_size], running_time_buff[time_buff_size];

    std::vector<double> loss_per_class(dataloader.num_classes(), 0);
    
    char message_buff[128];
    for(int step = 0; step < total_steps; step++){

        step_s_time = std::chrono::system_clock::now();

        BatchData batch = dataloader.get_batch(1);
        if(batch.input_data.size() == 0){
            // If the batch is empty, it means that the dataloader has reached the end of the dataset
            continue;
        }

        // we are sampling only one element at a time
        Tensor input = batch.input_data[0];
        Tensor target_one_hot = batch.targets_one_hot[0];
        int target = batch.targets_idx[0];

        Tensor output = forward(input);

        double *_output = output.data();
        double *_target_one_hot = target_one_hot.data();

        int output_size = output.size();

        int max_idx = 0;
        for(int i=0; i<output_size; i++){
            if(_output[max_idx] < _output[i]){
                max_idx = i;
            }
        }

        if(max_idx == target){
            correct++;
        }

        for(int i = 0; i < output_size; i++){
            tmp_loss = 0.5 * std::pow(_output[i] - _target_one_hot[i], 2);
            loss += tmp_loss;
            loss_per_class[i] += tmp_loss;
        }

        if(show_output){

            auto step_e_time = std::chrono::system_clock::now();

            std::chrono::duration<double> step_duration = step_e_time - step_s_time;
            std::chrono::duration<double> running_duration = step_e_time - running_s_time;

            make_duration_readable(step_duration, step_time_buff, time_buff_size);
            make_duration_readable(running_duration, running_time_buff, time_buff_size);

            std::sprintf(message_buff, "Loss: %.04f - Accuracy: %.04f - %s elapsed - %s/step", 
                loss/(step+1), (double)correct/(step+1), running_time_buff, step_time_buff);
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
    dataloader.new_epoch();

    return EvaluationResult{
        correct, total_steps, accuracy,
        loss/total_steps,
        loss_per_class};
}


void PlainNN::train(
    DataLoader& train_dataloader,
    double learning_rate,
    int epochs,
    int batch_size,
    bool save_checkpoint,
    std::string checkpoint_path
){
    _train(&train_dataloader, nullptr, learning_rate, epochs, batch_size, save_checkpoint, checkpoint_path);
}


void PlainNN::train(
    DataLoader& train_dataloader,
    DataLoader& test_dataloader,
    double learning_rate,
    int epochs,
    int batch_size,
    bool save_checkpoint,
    std::string checkpoint_path
){
    _train(&train_dataloader, &test_dataloader, learning_rate, epochs, batch_size, save_checkpoint, checkpoint_path);
}


void PlainNN::_train(
    DataLoader* train_dataloader,
    DataLoader* test_dataloader,
    double learning_rate,
    int epochs,
    int batch_size,
    bool save_checkpoint,
    std::string checkpoint_path
){
    if(save_checkpoint){
        if(checkpoint_path.empty()){
            std::printf("Checkpoint path is required to save the model\n");
            exit(1);
        }

        checkpoint_path = checkpoint_path + "_epoch_%d";
        
    }

    int steps_per_epoch = train_dataloader->steps_per_epoch(batch_size);
    
    char trailing_message_buff[128];
    size_t time_buff_size = 24;
    char epoch_running_time_buff[time_buff_size], step_time_buff[time_buff_size];

    for(int epoch=0; epoch < epochs; epoch++){

        auto epoch_s_time = std::chrono::system_clock::now();
        std::printf("Epoch %d/%d\n", epoch+1, epochs);

        for(int step = 0; step<steps_per_epoch; step++){

            auto step_s_time = std::chrono::system_clock::now();
            
            BatchData batch = train_dataloader->get_batch(batch_size);

            if(batch.input_data.size() == 0){
                // If the batch is empty, it means that the dataloader has reached the end of the dataset
                continue;
            }

            std::vector<Tensor> input = batch.input_data;
            std::vector<Tensor> batch_targets = batch.targets_one_hot;
            std::vector<int> batch_targets_idx = batch.targets_idx;

            double error = 0;
            int correct = 0;

            for(size_t b = 0; b<input.size(); b++){

                Tensor output = forward(input[b]);

                double *_output = output.data();
                double *_batch_targets = batch_targets[b].data();

                int output_size = output.size();

                for(int i=0; i<output_size; i++){
                    error += 0.5 * std::pow(_output[i] - _batch_targets[i], 2);
                }

                int max_idx = 0;
                for(int i=0; i<output_size; i++){
                    if(_output[max_idx] < _output[i]){
                        max_idx = i;
                    }
                }
                if(max_idx == batch_targets_idx[b]){
                    correct++;
                }
            
                int last_layer_idx = m_layers.size() - 1;
                Tensor next_layer_grads = Tensor();
                for(int layer_idx = last_layer_idx; layer_idx > 0; layer_idx--){
                    
                    if(m_layers[layer_idx]->is_frozen){
                        continue;
                    }

                    next_layer_grads = m_layers[layer_idx]->backward(
                        layer_idx == 1 ? &input[b] : &m_layers[layer_idx-1]->output,
                        layer_idx == last_layer_idx ? nullptr : m_layers[layer_idx+1]->get_params(),
                        layer_idx == last_layer_idx ? &batch_targets[b] : &next_layer_grads
                    );
                }
            }

            for(size_t layer_idx = 1; layer_idx < m_layers.size(); layer_idx++){
                if(m_layers[layer_idx]->is_frozen){
                    continue;
                }
                m_layers[layer_idx]->step(learning_rate, batch_size);
            }

            auto step_e_time = std::chrono::system_clock::now();
            std::chrono::duration<double> step_duration = step_e_time - step_s_time;
            std::chrono::duration<double> running_epoch_time = step_e_time - epoch_s_time;
            
            make_duration_readable(step_duration, step_time_buff, time_buff_size);
            make_duration_readable(running_epoch_time, epoch_running_time_buff, time_buff_size);

            std::sprintf(trailing_message_buff, "%s %s/step - Error: %.04f - Accuracy: %.04f", 
                epoch_running_time_buff, step_time_buff, error/batch_size, (double)correct/batch_size);
            print_progress(step+1, steps_per_epoch, trailing_message_buff, 20);
        }
        std::printf("\n");

        train_dataloader->new_epoch();

        if(test_dataloader != nullptr){
            evaluate(*test_dataloader, true, true);
        }

        if(m_lr_scheduler != nullptr){
            m_lr_scheduler->step(learning_rate, epoch);
        }

        if(save_checkpoint){
            char epoch_checkpoint_path[checkpoint_path.size() + std::to_string(epoch+1).size() + 1];
            std::sprintf(epoch_checkpoint_path, checkpoint_path.c_str(), epoch+1);
            save(epoch_checkpoint_path);
        }
    }
}


void PlainNN::count_to_size(int num_params, char* buff, size_t buff_size, size_t size){
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;

    double _num_params = static_cast<double>(num_params);
    if(size != 0)
        _num_params *= size;

    while(_num_params > 1024 && suffix_idx < 5){
        _num_params /= 1024;
        suffix_idx++;
    }

    std::snprintf(buff, buff_size, "%d (%.2f %s)", num_params, _num_params, suffixes[suffix_idx]);
}

void PlainNN::print_progress(int curr_progress, int total, std::string trailing_message, int width, bool indent){
    char progress_buff[50];
    int progress = (int)((curr_progress / (double)total) * width);
    for(int i = 0; i < width; i++){
        progress_buff[i] = i < progress ? '=' : ' ';
    }
    progress_buff[width] = '\0';
    std::printf("\r%s%d/%d [%s] %s", (indent) ? "    " : "", curr_progress, total, progress_buff, trailing_message.c_str());
    std::fflush(stdout);
}

void PlainNN::make_duration_readable(const std::chrono::duration<double>& duration, char* buff, size_t buff_size){
    
    long int us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    if(us >= 1000000){
        // If step duration is greater than 1 second,
        // print the time formatted in seconds
        std::snprintf(buff, buff_size, "%4lds", us/1000000);
    }else if (us >= 1000){
        // If step duration is greater than 1 millisecond,
        // print the time formatted in milliseconds
        std::snprintf(buff, buff_size, "%3ldms", us/1000);
    }else{
        // If step duration is less than 1 millisecond,
        // print the time formatted in microseconds
        std::snprintf(buff, buff_size, "%3ldus", us);
    }
}