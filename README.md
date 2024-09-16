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


## Uses 
- json.h from https://github.com/sheredom/json.h/tree/master
- stb_image_write.h and stb_image.h from https://github.com/nothings/stb/tree/master
