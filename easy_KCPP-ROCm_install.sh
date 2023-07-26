#!/bin/bash

git clone https://github.com/YellowRoseCx/koboldcpp-rocm.git -b main --depth 1 && \
cd koboldcpp-rocm && \
make LLAMA_HIPBLAS=1 -j4 && \
./koboldcpp.py

