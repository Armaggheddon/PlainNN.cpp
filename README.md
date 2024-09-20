## Compilation steps
Compile with
1. mkdir -p build
1. cd build
1. cmake ..
    optionally build with optimizations enable with 
    cmake -DCMAKE_BUILD_TYPE=Release .. 
1. make
1. ./mnist_cpp

## Dataset
Download the dataset from:
- https://github.com/fgnt/mnist

## Useful links
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

## Example of a train output on the MNIST dataset
```
___________________________________________________________
Layer        (Type)       Output Shape            Param #
===========================================================
input        (Input)      (784)                         0
dense        (Dense)      (128)                    100480
dense_1      (Dense)      (10)                       1290
===========================================================
Total params: 101770 (795.08 KB)
Trainable params: 101770 (795.08 KB)
Non-trainable params: 0 (0.00 B)
___________________________________________________________
Epoch 1/20
    937/937 [====================]   23s  24ms/step - Error: 0.1413 - Accuracy: 0.8750
Epoch 2/20
    937/937 [====================]   22s  24ms/step - Error: 0.0986 - Accuracy: 0.8906
Epoch 3/20
    937/937 [====================]   22s  24ms/step - Error: 0.0664 - Accuracy: 0.8906
Epoch 4/20
    937/937 [====================]   22s  24ms/step - Error: 0.0506 - Accuracy: 0.9531
Epoch 5/20
    937/937 [====================]   22s  24ms/step - Error: 0.0721 - Accuracy: 0.9219
Epoch 6/20
    937/937 [====================]   22s  24ms/step - Error: 0.0557 - Accuracy: 0.9688
Epoch 7/20
    937/937 [====================]   23s  24ms/step - Error: 0.0635 - Accuracy: 0.9219
Epoch 8/20
    937/937 [====================]   22s  24ms/step - Error: 0.0731 - Accuracy: 0.9062
Epoch 9/20
    937/937 [====================]   22s  24ms/step - Error: 0.0352 - Accuracy: 0.9844
Epoch 10/20
    937/937 [====================]   22s  24ms/step - Error: 0.0391 - Accuracy: 0.9688
Epoch 11/20
    937/937 [====================]   23s  25ms/step - Error: 0.0583 - Accuracy: 0.9375
Epoch 12/20
    937/937 [====================]   23s  25ms/step - Error: 0.0485 - Accuracy: 0.9375
Epoch 13/20
    937/937 [====================]   23s  24ms/step - Error: 0.0630 - Accuracy: 0.9375
Epoch 14/20
    937/937 [====================]   23s  24ms/step - Error: 0.0591 - Accuracy: 0.9219
Epoch 15/20
    937/937 [====================]   23s  24ms/step - Error: 0.0702 - Accuracy: 0.9375
Epoch 16/20
    937/937 [====================]   22s  24ms/step - Error: 0.0416 - Accuracy: 0.9375
Epoch 17/20
    937/937 [====================]   23s  25ms/step - Error: 0.0293 - Accuracy: 0.9844
Epoch 18/20
    937/937 [====================]   23s  24ms/step - Error: 0.0389 - Accuracy: 0.9688
Epoch 19/20
    937/937 [====================]   22s  24ms/step - Error: 0.0408 - Accuracy: 0.9531
Epoch 20/20
    937/937 [====================]   22s  24ms/step - Error: 0.0535 - Accuracy: 0.9219
10000/10000 [====================] Loss: 0.0418 - Accuracy: 0.9558 -    3s elapsed - 345us/step
Correct: 9558/10000
```

## Uses 
- json.h from https://github.com/sheredom/json.h/tree/master
- stb_image_write.h and stb_image.h from https://github.com/nothings/stb/tree/master
