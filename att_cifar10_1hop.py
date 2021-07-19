import os
import argparse
import logging
import time
import numpy as np
import scipy.linalg 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch.autograd import Function
import torch.jit
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1000) 
parser.add_argument('--save', type=str, default='./att_CIFAR10_1HOP') 
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epsilon', type=float, default = 0.01)
parser.add_argument('--tau', type=float, default = 1.0)
parser.add_argument('--run', type=int, default = 1)
args = parser.parse_args()


#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

# author: @nsde
# maintainer: @ferrine 
import torch.jit 
@torch.jit.script
def torch_pade13(A):  # pragma: no cover
    # avoid torch select operation and unpack coefs
    (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13) = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )

    ident = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b13 * A6 + b11 * A4 + b9 * A2)
        + b7 * A6
        + b5 * A4
        + b3 * A2
        + b1 * ident,
    )
    V = (
        torch.matmul(A6, b12 * A6 + b10 * A4 + b8 * A2)
        + b6 * A6
        + b4 * A4
        + b2 * A2
        + b0 * ident
    )
    return U, V

import torch.nn.functional as F
from torch.nn.modules.module import Module
 

@torch.jit.script
def matrix_2_power(x, p):  # pragma: no cover
    for _ in range(int(p)):
        x = x @ x
    return x


@torch.jit.script
def expm_one(A):  # pragma: no cover
    # no checks, this is private implementation
    # but A should be a matrix
    A_fro = torch.norm(A)

    # Scaling step

    n_squarings = torch.clamp(
        torch.ceil(torch.log(A_fro / 5.371920351148152).div(0.6931471805599453)), min=0
    )
    scaling = 2.0 ** n_squarings
    Ascaled = A / scaling

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V

    R, _ = torch.solve(P, Q)  # solve P = Q*R
    expmA = matrix_2_power(R, n_squarings)
    return expmA
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module): 
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
    


class NoiseBlock(nn.Module):
    
    def __init__(self, sigma):  
        super(NoiseBlock, self).__init__() 
        self.sigma = sigma 
        
    def forward(self, x): # sigma is noise power (standard deviation)
        out = x + self.sigma * torch.randn_like(x)
        return out
    
    def set_sigma(self, x):
        self.sigma = x
        return 1

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU()
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out
    
def concat_zero(x, additional_dim):
    B,C,H,W = x.size()
    zeros = torch.zeros(B,additional_dim,H,W).to(device)
    out = torch.cat((x,zeros),dim=1)
    return out

class concatBlock(nn.Module):
    
    def __init__(self):
        super(concatBlock, self).__init__()
        
    def forward(self, x):
        B,C,H,W = x.size()
        zeros = torch.zeros(B,6,H,W).to(device)
        out = torch.cat((x,zeros),dim=1)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc, t):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc 
        self.integration_time = torch.tensor([0, t]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)#, method='fixed_adams')
        #, method = 'tsit5')
        #, method = 'fixed_adams')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODEBlock_t(nn.Module):

    def __init__(self, odefunc, t):
        super(ODEBlock_t, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0,t]).float()
    
    def forward(self, x):
        self.odefunc.set_x0(x)
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='fixed_adams')
        return out[1] 
 

class ODEfunc_single_conv_nonConcat(nn.Module): # applies two convolution with the same weights

    def __init__(self, dim, N, epsilon): 
        super(ODEfunc_single_conv_nonConcat, self).__init__()
        self.dim = dim
        self.relu = nn.ReLU()
        module = nn.Conv2d 
        moduleT = nn.ConvTranspose2d
        self.conv1 = weight_norm( module(6+dim, 6+dim,  kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True) )
        self.Tconv1 = weight_norm( moduleT(6+dim, 6+dim,  kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True) )
        
        self.Tconv1.weight_v = self.conv1.weight_v
        self.Tconv1.weight_g = self.conv1.weight_g
        
        self.epsilon = epsilon
        self.nfe = 0 
        
        self.x0 = 0 
 
        self.conv_0 = weight_norm( module(6+dim, 6+dim,  kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True) )
        
        self.input_noisy = 0

        self.running_epsilon = 0
        #self.momentum = 0.1

        #self.norm02 = norm(dim+6)

    

    def forward(self, t, x):
        self.nfe += 1
        out_0 = self.x0 

        out = x
        out_e = x

        out_0 = self.relu(out_0)
        out_0 = self.conv_0(out_0)
        #out_0 = self.norm02(out_0)

        #out = self.relu(out)
        out = self.conv1(out)        
        out = self.relu(out)
        out = self.Tconv1(-out)
        #out = self.norm3(out)

        #out_e = self.norm1(out_e)
        out_e = -self.epsilon * torch.sum(self.conv_0.weight_g) * (self.dim + 6) * out_e
        #out_e = self.norm_e(out_e)
        
        out = out + out_e + out_0        

        self.running_epsilon = 0.9*self.running_epsilon + 0.1*self.epsilon*torch.sum(self.conv_0.weight_g.data)*(self.dim+6)

        return out
   
    def del_x0(self):
        del self.x0
 
    def set_x0(self,x0):
        self.x0 = x0

 
class ODEfunc_double_conv(nn.Module): # applies two convolution with the same weights

    def __init__(self, dim, N): 
        super(ODEfunc_double_conv, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU()
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        #self.conv2 = module(dim, dim,  kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True) 
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm3(out)
        return out
    
class ODEfunc_single_conv(nn.Module): # applies two convolution with the same weights

    def __init__(self, dim, N): 
        super(ODEfunc_single_conv, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU()
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        return out
    
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
def get_cifar10_loaders(data_aug=False, batch_size=128, test_batch_size = 1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def accuracy_reif(model, dataset_loader):
    total_correct = 0
    # original_sigma = model[].sigma
    # model[].set_sigma(0)
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10) 

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        
    # model[].set_sigma(original_sigma)
    return total_correct / len(dataset_loader.dataset)

def accuracy_reif_noisy(model, dataset_loader, sigma):
    total_correct = 0
    # original_sigma = model[].sigma
    # model[].set_sigma(0)
    for x, y in dataset_loader:
        x = x.to(device) + (torch.randn_like(x) * math.sqrt(sigma)).to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)

    # model[].set_sigma(original_sigma)
    return total_correct / len(dataset_loader.dataset)

def loss_reif(model,  dataset_loader, criterion):
    loss = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x) 

        loss = loss + criterion(logits, y)
     
    return loss 


def loss_reif_noisy(model, dataset_loader, sigma, criterion):
    loss = 0
    for x, y in dataset_loader:
        x = x.to(device) + (torch.randn_like(x) * math.sqrt(sigma)).to(device)
        y = y.to(device)
        logits = model(x) 

        loss = loss + criterion(logits, y)
     
    return loss 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger 



if __name__ == '__main__':

    args.save = args.save + str(args.tau) +"_sec_" + str(args.epsilon) + "_epsilon" +"_run"+ str(args.run)
    
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.propagate = False
    logger.info(args)  
    
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')   
    
    logger.info('designated gpu id by user : '+ str(args.gpu)) 
    if torch.cuda.is_available() :
        logger.info('GPU availability : True' )
    else :
        logger.info('\n!!!GPU NOT available!!!\n')
    #device = torch.device('cpu')
    is_odenet = args.network == 'odenet'
    
    NC = 16  # number of channels
    noise_power = 0.01
    reif_time = args.tau
    lambda_reif = 0.01
    epoch_start = 0
    epoch_end   = 240
    
    downsampling_layers = [
        nn.Conv2d(3, NC, 3, 1), # cifar10 : 32X32 -> 30X30  # mnist : 28X28 -> 26X26
        norm(NC),
        nn.ReLU(),
        nn.Conv2d(NC, NC, 4, 2, 1), # cifar10 : 30X30 -> 15X15  # mnist : 26X26 -> 13X13
        norm(NC),
        nn.ReLU(),
        nn.Conv2d(NC, NC, 4, 2, 1), # cifar10 : 15X15 -> 7X7  # mnist : 13X13 -> 6X6
    ]  
    
    concat_layer = [concatBlock()]

    feature_layers = [ODEBlock(ODEfunc(NC), reif_time)] 
    
    norm_layer_before_reif = [norm(NC)]
    
    reification_layers = [ODEBlock_t(ODEfunc_single_conv_nonConcat(dim = NC, N = 7, epsilon = args.epsilon), reif_time)] # 6 for mnist, 7 for cifar10 ###
    # 6 is correct for default setting, which is data_aug = TRUE. 
    
    fc_layers = [norm(NC+6), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(NC+6, 10)]

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_cifar10_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    
    model = nn.Sequential(*downsampling_layers, *feature_layers, *norm_layer_before_reif, *concat_layer, *reification_layers, *fc_layers).to(device)  
        
    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100 , 140, 200],
        decay_rates=[1, 0.5, 0.3, 0.1, 0.03]
    )

    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 
    
    best_acc = 0
    best_noisy_acc = 0
    best_ce_loss = math.inf
    best_ce_t_loss = math.inf
    best_sum_loss = math.inf
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    CE_loss_meter = RunningAverageMeter()
    CE_loss_t_meter = RunningAverageMeter()
    
    end = time.time() 
    
    for itr  in range(1+ epoch_start * batches_per_epoch,1+ epoch_end * batches_per_epoch):  

        for param_group in model_optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        model_optimizer.zero_grad()
        
        x, y = data_gen.__next__() 
        
        x = x.to(device)
        x_t = torch.add(x, math.sqrt(noise_power)*torch.randn_like(x).to(device))
        y = y.to(device) 
  
        '''
        state_without_noise = model(x)
        #state_without_noise_concat = concat_zero(state_without_noise, additional_dim = 6).to(device) 
        orig_state_reified = model_reif(state_without_noise)
        logits = model_fc( orig_state_reified )
        '''
        logits = model(x)
        CE_loss = criterion(logits, y)
        CE_loss.backward()
        
        '''
        state_noised = model(x_t)  
        #state_noised_concat = concat_zero(state_noised, additional_dim = 6).to(device) 
        state_reified = model_reif(state_noised)
        logits_t = model_fc( state_reified )
        '''
        logits_t = model(x_t)
        CE_loss_t = criterion(logits_t, y)
        CE_loss_t.backward()
        #REIF_loss = torch.norm(state_without_noise_concat[:,0:NC,:,:] - state_reified[:,0:NC,:,:], 2)/(128) #128 is batch size
        #REIF_loss = REIF_loss + torch.norm(state_without_noise_concat[:,0:NC,:,:] - orig_state_reified[:,0:NC,:,:], 2)/(128)
        #REIF_loss = REIF_loss / 2
        #loss = CE_loss +CE_loss_t 
        #loss_with_reif = CE_loss +CE_loss_t  + lambda_reif * REIF_loss /reif_time


        CE_loss_meter.update(CE_loss.data)
        CE_loss_t_meter.update(CE_loss_t.data)
        if is_odenet: #TODO : number of reification evaluation need to be counted
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0
        
        model_optimizer.step()   

        if is_odenet: #TODO : number of reification evaluation need to be counted
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time() 
        
        if itr %( batches_per_epoch) == 0:
            with torch.no_grad():  
                train_acc = accuracy_reif(model, train_eval_loader)
                val_acc = accuracy_reif(model, test_loader) 
                noisy_acc = accuracy_reif_noisy(model, test_loader, noise_power) 
                vloss = loss_reif(model, test_loader, criterion) 
                vtloss = loss_reif_noisy(model, test_loader, noise_power, criterion) 
                if val_acc > best_acc :
                    torch.save(model, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc  
                  
                if noisy_acc > best_noisy_acc : 
                    torch.save(model, os.path.join(args.save, 'model_noisy.pth'))
                    best_noisy_acc = noisy_acc 
                    
                if vloss < best_ce_loss :
                    torch.save(model, os.path.join(args.save, 'model_vloss.pth'))
                    best_ce_loss = vloss
                
                if vtloss < best_ce_t_loss :
                    torch.save(model, os.path.join(args.save, 'model_vtloss.pth'))
                    best_ce_t_loss = vtloss
                
                if (vloss + vtloss) < best_sum_loss : 
                    torch.save(model, os.path.join(args.save, 'model_sumloss.pth'))
                    best_sum_loss = vloss + vtloss

                logger.info(
                    "Epoch {:04d} | Time ({:.3f}) | "
                    "Train Acc {:.4f} | Test Acc {:.4f} | Noisy Acc {:.4f}| CEloss {:.4f} | CEtloss {:.4f} | "
                    "sumloss {:.4f}| vloss {:.4f} | vtloss {:.4f} | vsumloss {:.4f} | running_eps {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.avg,
                        train_acc, val_acc, noisy_acc, CE_loss_meter.avg, CE_loss_t_meter.avg, CE_loss_meter.avg+CE_loss_t_meter.avg,
                        vloss, vtloss, vloss + vtloss, model[10].odefunc.running_epsilon
                    )
                    
                )
