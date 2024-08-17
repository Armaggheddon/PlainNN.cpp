#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <string>


typedef void (*act_func)(std::vector<double> *) ;

typedef struct{
    std::string layer_name;
    int input_size, batch_size, output_size;
    int param_count;
    long int param_size; // in bytes
} LayerSummary;

class Layer{
    public:
        std::vector<std::vector<double> > output;
        std::vector<std::vector<double> > grad;
        virtual ~Layer(){};
        virtual void initialize(int input_size) = 0;
        virtual void forward(std::vector<std::vector<double> > *input) = 0;
        virtual void backward(std::vector<std::vector<double> > *input) = 0;
        virtual LayerSummary get_summary() = 0;
};

class Dense: public Layer{
    public:
        Dense(int output_size, act_func activation = NULL);
        virtual void initialize(int input_size);
        virtual void forward(std::vector<std::vector<double> > *input);
        virtual void backward(std::vector<std::vector<double> > *input);
        virtual LayerSummary get_summary();

    private: 
        act_func activation;
        std::vector<std::vector<double> > weights;
        std::vector<double> bias;
        LayerSummary summary;
        bool initialized = false;
};

class Input: public Layer{
    public: 
        Input(int output_size);
        virtual void initialize(int input_size);
        virtual void forward(std::vector<std::vector<double> > *input);
        virtual void backward(std::vector<std::vector<double> > *input);
        virtual LayerSummary get_summary();
    private:
        LayerSummary summary;
        bool initialized = false;        
};

#endif