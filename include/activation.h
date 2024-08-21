#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>


class ActivationFn{
    public:
        virtual void forward(std::vector<float> *matrix) = 0;
        virtual void backward(std::vector<float> *matrix) = 0;
};

class ReLU : public ActivationFn{
    public:
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

class Sigmoid : public ActivationFn{
    public:
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

class Softmax : public ActivationFn{
    public:
        void forward(std::vector<float> *matrix);
        void backward(std::vector<float> *matrix);
};

#endif // ACTIVATION_H