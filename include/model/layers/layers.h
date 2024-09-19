#ifndef MODEL_LAYERS_LAYERS_H
#define MODEL_LAYERS_LAYERS_H

#include <vector>

#include "tensor.h"
#include "activation_fncs.h"

/**
 * @brief Enum to hold the type of the layer
 */
enum LayerType{
    INPUT,
    DENSE
};

/**
 * @brief Array of layer type names
 */
const std::string LAYER_TYPE_NAMES[] = {
    "Input",
    "Dense"
};

/**
 * @brief Struct to hold the summary of a layer
 */
struct LayerSummary{
    LayerType layer_type;
    std::string layer_name;
    std::string activation_fn;
    int param_count;
    long int param_size;
    std::vector<int> layer_shape;
};

/**
 * @brief Abstract class for a layer in a neural network.
 * New layers should inherit from this class and implement
 * all of its methods.
 */
class Layer{
    public:

        bool is_initialized = false;
        Tensor output;
        LayerType layer_type;

        bool is_frozen = false;

        ~Layer(){};

        /**
         * @brief Forward pass of the layer
         * 
         * @param input The input to the layer
         * @return Tensor The output of the layer
         */
        virtual Tensor& forward(Tensor& input) = 0;
        
        /**
         * @brief Backward pass of the layer
         * 
         * @param prev_output The output of the previous layer
         * @param next_weights The weights of the next layer
         * @param next_grad The gradient of the next layer
         * 
         * @return Tensor The gradient of the layer
         * 
         * @note If the layer is frozen this function will
         * never be called. If the layer is an output layer
         * next_weights will be a null pointer and next_grad 
         * will be the gradient of the loss function.
         */
        virtual Tensor backward(Tensor*  prev_output, Tensor* next_weights, Tensor* next_grad) = 0;
        
        /**
         * @brief Update the weights of the layer,
         * should be called after the epoch is done.
         * 
         * @param learning_rate The learning rate
         * @param batch_size The batch size
         */
        virtual void step(double learning_rate, int batch_size) = 0;
        
        /**
         * @brief Get the saveable parameters of the layer
         * 
         * @return std::vector<double> The saveable parameters
         * 
         * @note This is used to save the model to a file. Each 
         * implementation should flatten the layer's parameters into
         * single a one dimensional vector.
         */
        virtual std::vector<double> get_saveable_params() = 0;

        /**
         * @brief Load the parameters of the layer
         * 
         * @param params The parameters to load
         * 
         * @note This is used to load the model from a file. Each 
         * implementation should load the parameters from a one 
         * dimensional vector and load them according to the internal
         * parameter layout.
         */
        virtual void load_params(std::vector<double>& params) = 0;

        /**
         * @brief Get the parameters of the layer
         * 
         * @return Tensor* The parameters of the layer
         * 
         * @note This is used during backpropagation to get
         * the parameters to pass to the layer in order to 
         * calculate the gradients. 
         */
        virtual Tensor* get_params(){return new Tensor();};

        /**
         * @brief Initialize the layer
         * 
         * @param input_shape The shape of the input to the layer
         * 
         * @note This is used to initialize the layer with the 
         * correct shape. This is called before the first forward
         * pass of the layer.
         */
        virtual void initialize(std::vector<int> input_shape) = 0;

        /**
         * @brief Get the summary of the layer
         * 
         * @return LayerSummary The summary of the layer
         * 
         * @note This is used to get a summary of the layer
         * to print to the console.
         */
        virtual LayerSummary get_summary() = 0;

        /**
         * @brief Freeze the layer
         * 
         * @param freeze Whether to freeze the layer
         * 
         * @note This is used to freeze the layer so that
         * the weights are not updated during training.
         */
        void freeze(bool freeze = true);

        /**
         * @brief Get the name of the layer
         * 
         * @return std::string The name of the layer
         * 
         */
        std::string name();
};
/**
 * @brief Build a layer from the name
 * 
 * @param name The name of the layer
 * @param layer_shape The shape of the layer
 * @param activation_fn The activation function of the layer
 * 
 * @return Layer* The layer object
 */
Layer* build_layer_from_name(std::string name, std::vector<int> layer_shape, ActivationFn* activation_fn);

/**
 * @brief Input layer, this layer has no parameters
 * and is only used to pass the input to the other layers.
 */
class Input : public Layer{
    public:
        Input(std::vector<int> shape, bool frozen = false);
        Input(std::initializer_list<int> shape, bool frozen = false);

        Tensor& forward( Tensor& input);
        Tensor backward( Tensor* prev_output,  Tensor* next_weights,  Tensor* next_grad);
        void step(double learning_rate, int batch_size);
        std::vector<double> get_saveable_params();
        void load_params( std::vector<double>& params);

        void initialize(std::vector<int> input_shape);

        LayerSummary get_summary();
};

/**
 * @brief Dense layer is a fully connected layer
 * with an activation function. It has two parameters
 * weights and biases. 
 */
class Dense : public Layer{
    public:
        Dense(int output_size, ActivationFn* activation_fn, bool frozen = false);
        Dense(int input_size, int output_size, ActivationFn* activation_fn, bool frozen = false);

        void initialize(std::vector<int> input_shape);
        Tensor* get_params() override;
        Tensor& forward(Tensor& input);
        Tensor backward(Tensor* prev_output, Tensor* next_weights, Tensor* next_grad);
        void step(double learning_rate, int batch_size);
        std::vector<double> get_saveable_params();
        void load_params( std::vector<double>& params);

        LayerSummary get_summary();

    private:
        int input_size, output_size; 
        Tensor weights;
        Tensor d_weights;
        Tensor biases;
        Tensor d_biases;
        ActivationFn* activation_fn;
};

#endif // MODEL_LAYERS_LAYERS_H