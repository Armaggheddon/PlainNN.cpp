#include "model.h"

#include "layers.h"
#include "data_loaders.h"

#include <vector>
#include <chrono>
#include <algorithm>

Model::Model(){}

void Model::add_layer(Layer* layer){
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


Layer* Model::get_layer(int index){
    return m_layers[index];
}


void Model::set_lr_scheduler(LRScheduler* scheduler){
    m_lr_scheduler = scheduler;
}


// TODO: Implement this function
void Model::summary(){
    std::printf("_________________________________________________________________\n");
    std::printf("Layer (type)                 Output Shape              Param #\n");
    std::printf("=================================================================\n");

    // int total_params = 0;

    // for(int i = 0; i < layers.size(); i++){
    //     Layer* layer = layers[i];
    //     std::string layer_name = "[" + std::to_string(i) + "] " + layer->name();
    //     std::string output_shape = "( 1, " + std::to_string(layer->config.output_size) + ")";
    //     int num_params = 0;

    //     if(layer->config.type == LayerType::DENSE){
    //         Dense* dense_layer = dynamic_cast<Dense*>(layer);
    //         if(dense_layer == nullptr){
    //             std::printf("Error during cast from Layer to Dense\n");
    //             return;
    //         }
    //         num_params = dense_layer->config.input_size * dense_layer->config.output_size + dense_layer->config.output_size;
    //     }

    //     total_params += num_params;

    //     std::printf("%-24s %-24s %12d\n", layer_name.c_str(), output_shape.c_str(), num_params);
    // }

    // std::printf("=================================================================\n");

    // char trainable_params_buff[16], non_trainable_params_buff[16], total_params_buff[16];
    // num_params_to_size(total_params * sizeof(double), total_params_buff);
    // num_params_to_size(total_params * sizeof(double), trainable_params_buff);
    // num_params_to_size(0 * sizeof(double), non_trainable_params_buff);

    // std::printf("Total params: %d (%s)\n", total_params, total_params_buff);
    // std::printf("Trainable params: %d (%s)\n", total_params, trainable_params_buff);
    // std::printf("Non-trainable params: 0 (%s)\n", non_trainable_params_buff);
    // std::printf("_________________________________________________________________\n");
}


Tensor Model::forward(Tensor& input){
    Tensor output = input;

    for(int i = 1; i < m_layers.size(); i++){
        output = m_layers[i]->forward(output);
    }

    return output;
}


EvaluationResult Model::evaluate(DataLoader& dataloader, bool show_output, bool indent){
    int correct = 0, total = 0;
    int total_steps = dataloader.steps_per_epoch(1);
    double accuracy = 0, loss = 0, tmp_loss = 0;

    auto running_s_time = std::chrono::system_clock::now();
    auto step_s_time = std::chrono::system_clock::now();

    char step_time_buff[16], running_time_buff[16];

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

            make_duration_readable(step_duration, step_time_buff);
            make_duration_readable(running_duration, running_time_buff);

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


void Model::train(
    DataLoader& train_dataloader,
    double learning_rate,
    int epochs,
    int batch_size
){
    _train(&train_dataloader, nullptr, learning_rate, epochs, batch_size);
}


void Model::train(
    DataLoader& train_dataloader,
    DataLoader& test_dataloader,
    double learning_rate,
    int epochs,
    int batch_size
){
    _train(&train_dataloader, &test_dataloader, learning_rate, epochs, batch_size);
}


void Model::_train(
    DataLoader* train_dataloader,
    DataLoader* test_dataloader,
    double learning_rate,
    int epochs,
    int batch_size
){
    int steps_per_epoch = train_dataloader->steps_per_epoch(batch_size);
    
    char trailing_message_buff[128];
    char training_running_time[16], epoch_running_time_buff[16], step_time_buff[16];

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

            for(int b = 0; b<input.size(); b++){

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
                for(int layer = last_layer_idx; layer > 0; layer--){

                    next_layer_grads = m_layers[layer]->backward(
                        layer == 1 ? &input[b] : &m_layers[layer-1]->output,
                        layer == last_layer_idx ? nullptr : m_layers[layer+1]->get_params(),
                        layer == last_layer_idx ? &batch_targets[b] : &next_layer_grads
                    );
                }
            }

            for(int layer = 1; layer < m_layers.size(); layer++){
                m_layers[layer]->step(learning_rate, batch_size);
            }

            auto step_e_time = std::chrono::system_clock::now();
            std::chrono::duration<double> step_duration = step_e_time - step_s_time;
            std::chrono::duration<double> running_epoch_time = step_e_time - epoch_s_time;
            
            make_duration_readable(step_duration, step_time_buff);
            make_duration_readable(running_epoch_time, epoch_running_time_buff);

            std::sprintf(trailing_message_buff, "%s %s/step - Error: %.04f - Accuracy: %.04f", 
                epoch_running_time_buff, step_time_buff, error/batch_size, (double)correct/batch_size);
            print_progress(step, steps_per_epoch, trailing_message_buff, 20);
        }
        std::printf("\n");

        train_dataloader->new_epoch();

        if(test_dataloader != nullptr){
            evaluate(*test_dataloader, true, true);
        }

        if(m_lr_scheduler != nullptr){
            m_lr_scheduler->step(learning_rate, epoch);
        }
    }
}


void Model::count_to_size(int num_params, char* buff, size_t size){
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;

    num_params *= size;

    while(num_params > 1024 && suffix_idx < 5){
        num_params /= 1024;
        suffix_idx++;
    }

    std::snprintf(buff, size, "%s", suffixes[suffix_idx]);
}

void Model::print_progress(int curr_progress, int total, std::string trailing_message, int width, bool indent){
    char progress_buff[50];
    int progress = (int)((curr_progress / (double)total) * width);
    for(int i = 0; i < width; i++){
        progress_buff[i] = i < progress ? '=' : ' ';
    }
    progress_buff[width] = '\0';
    std::printf("\r%s%d/%d [%s] %s", (indent) ? "    " : "", curr_progress, total, progress_buff, trailing_message.c_str());
    std::fflush(stdout);
}

void Model::make_duration_readable(const std::chrono::duration<double>& duration, char* buff){
    
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