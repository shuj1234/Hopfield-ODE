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

Above training codes are written based on 'https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py' from 'torchdiffeq'

arg --epsilon determines the value of epsilon hyperparameter of proposed ODE layers. Usage example : python att_cifar10_1hop.py --epsilon 0.01

arg --tau determines the integration time of ODE layers (including both conventional and proposed ODE layers). Usage example : python att_cifar10_2hop.py --tau 20.0

arg --run takes integer input and is attached in str form in the saving directory. If one runs python att_SVHN_1hop.py --run 1, then the trained model will be saved in the directory ./att_SVHN_2HOP_1.0_sec_0.01_epsilon_run_1/
