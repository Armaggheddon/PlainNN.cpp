#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <string>

class ActivationFn{
    public:
        std::string name;
        virtual void forward(std::vector<float> *matrix) = 0;
        virtual void backward(std::vector<float> *matrix) = 0;
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

#endif // ACTIVATION_H