# Hopfield-ODE
Code reveal for paper under review, 'Hopfield-type Neural Ordinary Differential Equation for Robust Machine Learning'

run att_cifar10_2hop.py to train 'Proposed' setting on cifar-10 dataset. (two proposed layers)
run att_cifar10_1hop.py to train 'Mixed' setting on cifar-10 dataset. (one conventional ODE layer, one proposed ODE layer)
run att_cifar10_NOhop.py to train 'Conv_only' setting on cifar-10 dataset. (two conventional ODE layers)
run discrete_conv_cifar10.py to train 'Discrete Convolution' network on cifar-10 dataset. (Non-ODE discrete layers)

run att_SVHN_2hop.py to train 'Proposed' setting on SVHN dataset. 
run att_SVHN_1hop.py to train 'Mixed' setting on cifar-10 dataset. (one conventional ODE layer, one proposed ODE layer)
run att_SVHN_NOhop.py to train 'Conv_only' setting on cifar-10 dataset. (two conventional ODE layers)
run discrete_conv_SVHN.py to train 'Discrete Convolution' network on cifar-10 dataset. (Non-ODE discrete layers)

Above training codes are modifications of 'https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py' from 'torchdiffeq'

