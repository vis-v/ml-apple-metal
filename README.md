# ml-apple-metal
Trying to do some Machine Learning on an Apple Silicon Machine? This repo helps you to solve some issues in transitioning from x86 to ARM. ANE is utilized for from core ML.

Tensorflow Setup: https://developer.apple.com/metal/tensorflow-plugin/

PyTorch Setup: https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

python3 check-ml-versions.py

Library              Version
---------------------------
Python Platform      macOS-12.3-arm64-arm-64bit
Python               3.9.7
TensorFlow Version   2.7.0
Keras Version        2.7.0
Pandas               1.3.3
Scikit-Learn         1.0
PyTorch              1.12.0

TensorFlow GPU support: available
Note: 'available' indicates TensorFlow is using GPU.

PyTorch MPS support: available
MPS device found: tensor([1.], device='mps:0')
