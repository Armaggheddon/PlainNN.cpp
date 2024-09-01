#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <string>

enum ActivationFnType{
    RELU = 0, 
    SIGMOID = 1,
    SOFTMAX = 2,
    NONE = 3
};

const std::string ACTIVATION_FN_NAMES[] = {
    "relu",
    "sigmoid",
    "softmax",
    "none"
};


class ActivationFn{
    public:
        virtual void forward(std::vector<float> *matrix) = 0;
        virtual void backward(std::vector<float> *matrix) = 0;
        virtual std::string get_name(){ return ACTIVATION_FN_NAMES[type]; }
    protected:
        ActivationFnType type;
};

class ReLU : public ActivationFn{
    public:
        ReLU();
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

class Sigmoid : public ActivationFn{
    public:
        Sigmoid();
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

class Softmax : public ActivationFn{
    public:
        Softmax();
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

class None : public ActivationFn{
    public:
        None();
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

#endif // ACTIVATION_H