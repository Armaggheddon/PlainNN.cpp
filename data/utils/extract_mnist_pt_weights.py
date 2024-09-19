import pathlib
import struct
import enum

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class DType(enum.Enum):
    DOUBLE = "double"

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
def extract_weights(model_path):
    """
    Export the weights of a PyTorch model to a binary file named
    after the model file with the extension changed to .weights
    and with the dtype appended to the name.

    Args:
    - model_path (str): The path to the PyTorch model file.
    - dtype (DType): The data type to convert the weights to.
    """
    model = MNIST()
    model.load_state_dict(torch.load(model_path))

    weights_path = pathlib.Path(model_path).with_suffix(".weights")
    new_path = weights_path.parent / (weights_path.stem + ".weights")

    # Extract weights
    fc1_weight = model.fc1.weight.data.numpy().astype(np.double).T
    fc1_bias = model.fc1.bias.data.numpy().astype(np.double)

    fc2_weight = model.fc2.weight.data.numpy().astype(np.double).T
    fc2_bias = model.fc2.bias.data.numpy().astype(np.double)

    print(f"Extracting weights from {model_path} to {new_path} as double...")

    dtype_t = "d"

    with open(new_path, "wb") as f:
        print(f"\tFC1 weight shape: {fc1_weight.shape}, type: {fc1_weight.dtype}")
        ffc1 = fc1_weight.flatten(order="C") # flatten in column-major order
        for i in range(ffc1.shape[0]):
                f.write(struct.pack(f"{dtype_t}", ffc1[i]))
        print(f"\tFC1 bias shape: {fc1_bias.shape}, type: {fc1_bias.dtype}")
        for i in range(fc1_bias.shape[0]):
            f.write(struct.pack(f"{dtype_t}", fc1_bias[i]))
        print(f"\tFC2 weight shape: {fc2_weight.shape}, type: {fc2_weight.dtype}")
        ffc2 = fc2_weight.flatten(order="C") # flatten in column-major order
        for i in range(ffc2.shape[0]):
            f.write(struct.pack(f"{dtype_t}", ffc2[i]))
        print(f"\tFC2 bias shape: {fc2_bias.shape}, type: {fc2_bias.dtype}")
        for i in range(fc2_bias.shape[0]):
            f.write(struct.pack(f"{dtype_t}", fc2_bias[i]))

def check_weights_same( layer_name, i_max, j_max, i, j):
    """
    Check if the weights extracted from the PyTorch model are the same
    as the weights extracted from the binary file. More of a sanity check.

    Args:
    - layer_name (str): The name of the layer to check.
    - i_max (int): The number of rows in the weight matrix, i.e. number of output neurons of the layer.
    - j_max (int): The number of columns in the weight matrix, i.e. number of input neurons of the layer.
    - i (int): The row index to check.
    - j (int): The column index to check.
    - dtype (DType): The data type of the weights.
    """
    model = MNIST()
    model.load_state_dict(torch.load("mnist_fc128_relu_fc10_sigmoid.pt"))

    layer = getattr(model, layer_name)
    weight = layer.weight.data.numpy()
    bias = layer.bias.data.numpy()

    # load from file
    weights_path = pathlib.Path(f"mnist_fc128_relu_fc10_sigmoid.weights")
    dtype_t = "d"
    size_t = 8 # 8 bytes per double

    with open(weights_path, "rb") as f:
        seek_to = i * j_max + j # seek to the correct weight
        weights_count = j_max*i_max # total number of weights
        f.seek(size_t * seek_to)
        f_weight = struct.unpack(f"{dtype_t}", f.read(size_t))[0]
        f.seek(0)
        f.seek(size_t * (weights_count + i))
        f_bias = struct.unpack(f"{dtype_t}", f.read(size_t))[0]

    # Print a table to compare the weights
    print()
    print(f"{f'Layer {layer_name} as double':<20} | {'PyTorch':<25} | {'File':<25}")
    print(f"{'-'*20}-|-{'-'*25}-|-{'-'*25}")
    print(f"{f'weight[{i}][{j}]':<20} | {weight[i][j]:<25.20} | {f_weight:<25.20}")
    print(f"{f'bias[{i}]':<20} | {bias[i]:<25.20} | {f_bias:<25.20}")
    print()

if __name__ == '__main__':
    model_path = "mnist_fc128_relu_fc10_sigmoid.pt"
    extract_weights(model_path)
    
    # check_weights_same("fc1", 128, 784, 127, 783, DType.DOUBLE)