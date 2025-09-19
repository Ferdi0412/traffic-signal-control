"""

Functions
---------
 - get_cpu()
 - get_gpu([idx])
 - get_dataloader([batch_size], [train], [seed])
 - train(cls, dloader, optim_cls, *, [device], [seed], [**kwargs])
 - test(net, dloader)
 - test_classes(net, dloader)
 - save(net, [acc])
 """
import inspect, datetime

import torch, torchvision

import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

def get_cpu() -> torch.device:
    """Simply returns the CPU device for use with PyTorch."""
    return torch.device('cpu')

def get_gpu(idx: int=0) -> torch.device:
    """Will return the GPU device for use with PyTorch if possible."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{idx}")
    else:
        print("Warning: CUDA is not available - using CPU!")
        return get_cpu()

def get_dataloader(batch_size: int, train: bool=True, seed: int=0, **kwargs):
    """Get a dataloader with the desired batch size. Default is training dataset.
    
    Example
    -------
    >> trainloader = get_dataloader(4, True)
    >> testloader  = get_dataloader(4, False)
    """
    torch.manual_seed(seed)
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=train,
                                           download=True, transform=t)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=train, num_workers=2,
                                       **kwargs)

def train(cls, dloader, optim_cls, *, device=None, seed=0, get_losses=False, **kwargs):
    """Trains and returns the desired class.
    
    Params
    ------
    cls : type (torch.NN.Module)
        This is a class (not an instance) that defines the neural net
    dloader : torch.utils.data.DataLoader
        This is what get_dataloader returns.
    optim_cls : type (torch.optim.Optimizer)
        This is a class (not an instance) to be used for the optimizer
    [The following params must be defined by keyword]
    device : torch.device
        Use get_gpu() if you want to use GPU, defaults to get_cpu()
    seed : int
        To make every run of the same model consistent
    silent : bool
        To drop the loss calculation - should make it faster
    get_losses : bool
        Return the losses
    **kwargs
        Any argument here will be used to create the desired optimizer

    Example
    -------
    >> class Net:
    >>     def __init__():
    >>         ...
    >> net = train(Net, get_dataloader(4), optim.SGD, lr=0.004, momentum=0.9)
    """
    torch.manual_seed(seed)

    if device is None:
        device = get_cpu()

    criterion = nn.CrossEntropyLoss()
    net = cls().to(device)

    optimizer = optim_cls(net.parameters(), **kwargs)
    losses = []
    
    for epoch in range(2):
        for i, d in enumerate(dloader, 0):
            ins, labels = d[0].to(device), d[1].to(device)

            optimizer.zero_grad()

            outs = net(ins)

            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            
            if get_losses:
                losses.append(loss.item())
                
    if get_losses:
      return net, losses  
    return net

def save_model(net, *, dir: str=None, acc: float=None, name: str=None, drop_date: bool=False):
    """Saves the model based on test accuracy, structure, and date.
    
    Note
    ----
    The name will be formatted:
        [{acc}p_][{name}_]yyyy_mm_dd_hh_mm_ss.pth

    Example
    -------
    >> save_model(net, acc=50.) # "50p_2025_09_15_09_13_40.pth
    >> save_model(net, name="Best Performance") # Best Performance_2025_09_15_09_13_40.pth
    >> save_model(net) # 2025_09_15_09_13_40.pth
    """
    name  = f"{dir}/" if dir else "" 
    name += f"{acc}p_" if acc else ""
    name += f"{name}_" if name else ""
    name += datetime.datetime.now().strftime("%Y_%m%d %H:%M:%S") if not drop_date else ""
    if not name:
        raise ValueError("File name is empty! Can't have that!")
    name += ".pth"
    torch.save(net.statedict(), name)
    return name

def test(net, dloader, device=None):
    """Test a trained model.
    
    Example
    -------
    >> acc = test(net, testloader)
    """
    correct = 0
    total   = 0
    with torch.no_grad():
        for d in dloader:
            if device:
                images = d[0].to(device)
                labels = d[1].to(device)
            else:
                images, labels = d
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def calc_out_shape(shape, conv):
    """Caclulate the output shape after a single convolution OR pooling step.

    After Convolution:
    DIM_OUT = (DIM_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    Example
    -------
    >> calc_conv_out_shape(32, nn.Conv2d(3, 6, 5))
    """
    shape = (shape, shape) if isinstance(shape, int) else shape
    padding = conv.get('padding', 0) if isinstance(conv, dict) else conv.padding
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = conv.get('dilation', 1) if isinstance(conv, dict) else conv.dilation if hasattr(conv, "dilation") else 1
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    kernel_size = conv['kernel_size'] if isinstance(conv, dict) else conv.kernel_size
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = conv.get('stride', 1) if isinstance(conv, dict) else conv.stride
    stride = (stride, stride) if isinstance(stride, int) else stride
    return int((shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1),\
           int((shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)

def make_Net(convs, lins, *, pool=None, activ=F.relu, bnorm=True, drop_softmax=False):
    """
    
    Params
    ------
    convs : ...
    lins : list (int)
        Hidden layer sizes
    pool : nn.Module
        The pool to apply
    activ : function
        The activation function to apply.
    bnorm : bool
        Whether to apply batch nomalization
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            self.convs, self.bncs = nn.ModuleList(), nn.ModuleList()
            self.lins, self.bnls = nn.ModuleList(), nn.ModuleList()
            self.activ = activ
            self.pool = pool
            self.drop_softmax = drop_softmax
            
            channels = 3
            img_shape = (32, 32)
            for c in convs:
                if isinstance(c, dict):
                    self.convs.append(nn.Conv2d(channels, **c))
                else:
                    self.convs.append(nn.Conv2d(channels, *c))
                # TODO Update img_shape
                channels = self.convs[-1].out_channels
                img_shape = calc_out_shape(img_shape, self.convs[-1])
                if bnorm:
                    self.bncs.append(nn.BatchNorm2d(channels))
                if self.pool:
                    img_shape = calc_out_shape(img_shape, self.pool)
            vec_size = img_shape[0] * img_shape[1] * channels
            for l in lins:
                self.lins.append(nn.Linear(vec_size, l))
                vec_size = l
                if bnorm:
                    self.bnls.append(nn.BatchNorm1d(vec_size))
            self.last_layer = nn.Linear(vec_size, 10)

        def forward(self, x):
            ## Apply Convolutional layers
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if self.bncs:
                    x = self.bncs[i](x)
                x = self.activ(x)
                if self.pool:
                    x = self.pool(x)
            # Apply Linear layers
            x = torch.flatten(x, 1)
            for i, lin in enumerate(self.lins):
                x = lin(x)
                if self.bnls:
                    x = self.bnls[i](x)
                x = self.activ(x)
            if self.drop_softmax:
                return self.last_layer(x)
            else:
                return F.softmax(self.last_layer(x))
    return Net