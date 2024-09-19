#ifndef MODEL_LAYERS_ACTIVATION_FNCS_H
#define MODEL_LAYERS_ACTIVATION_FNCS_H

#include <cmath>
#include <string>

enum ActivationType{
    NONE,
    RELU,
    SIGMOID,
    TANH
};

const std::string ACTIVATION_NAMES[] = {
    "None",
    "ReLU",
    "Sigmoid",
    "Tanh"
};

/**
 * @brief Abstract class for activation functions, new
 * activation functions should inherit from this class
 * and implement all of its methods
 */
class ActivationFn{
    public:
        ~ActivationFn(){};

        ActivationType fn_type;

        /**
         * @brief Forward pass of the activation function
         * 
         * @param input The input to the activation function
         * @return double The output of the activation function
         */
        virtual double forward(const double input) = 0;

        /**
         * @brief Backward pass of the activation function
         * 
         * @param input The input to the activation function
         * @return double The gradient of the activation function
         */
        virtual double backward(const double input) = 0;

        /**
         * @brief Get the name of the activation function
         * 
         * @return std::string The name of the activation function
         */
        std::string name(){return ACTIVATION_NAMES[fn_type];}

        /**
         * @brief Get the type of the activation function
         * 
         * @return ActivationType The type of the activation function
         */
        ActivationType type(){return fn_type;}
};
/**
 * @brief Get the activation function object from the name
 * 
 * @param name The name of the activation function
 * @return ActivationFn* The activation function object
 */
ActivationFn* get_activation_fn_from_name(std::string name);


/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * 
 * f(x) = max(0, x)
 * 
 * f'(x) = 1 if x > 0, 0 otherwise
 */
class ReLU : public ActivationFn{
    public:
        ReLU();

        double forward(const double input);

        double backward(const double input);
};

/**
 * @brief Sigmoid activation function
 * 
 * f(x) = 1 / (1 + exp(-x))
 * 
 * f'(x) = f(x) * (1 - f(x))
 */
class Sigmoid : public ActivationFn{
    public:
        
        Sigmoid();

        double forward(const double input);

        double backward(const double input);
};

/**
 * @brief Hyperbolic Tangent (Tanh) activation function
 * 
 * f(x) = tanh(x)
 * 
 * f'(x) = 1 - f(x)^2
 */
class Tanh : public ActivationFn{
    public:
        
        Tanh();

        double forward(const double input);

        double backward(const double input);
};

class None : public ActivationFn{
    public:
        /**
         * @brief No activation function
         * 
         * f(x) = x
         * 
         * f'(x) = 1
         */
        None();

        double forward(const double input);

        double backward(const double input);
};

#endif // MODEL_LAYERS_ACTIVATION_FNCS_H