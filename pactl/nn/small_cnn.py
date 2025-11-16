import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models import register_model
from collections import OrderedDict


class Named(type):
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return self.__name__


class ConvBNrelu(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class layer13s(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10, in_chans=3, base_width=8):
        super().__init__()
        k = 2*base_width
        self.num_classes = num_classes
        self.net = nn.Sequential(
            ConvBNrelu(in_chans,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
        )
        self.fc = nn.Linear(2*k,num_classes)
    def forward(self,x):
        return self.fc(self.net(x))

class Expression(nn.Module):
    def __init__(self,func):
        super().__init__()
        self.func = func
    def forward(self,x):
        return self.func(x)


# Define default layers functions
def linear_layer(in_dim, out_dim, use_bias=True):
  return nn.Linear(in_dim, out_dim, use_bias)

def conv2d_layer(in_channels, out_channels, kernel_size, use_bias=True, stride=1, padding=0, dilation=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(FcNet3, self).__init__()
        self.layers_names = ('FC1', 'FC2', 'FC3', 'FC_out')
        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 400
        n_hidden3 = 400
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc3 = linear_layer(n_hidden2, n_hidden3)
        self.fc_out = linear_layer(n_hidden3, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        x = x.view(-1, self.input_size)  # flatten image
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc_out(x)
        return x


def get_size_of_conv_output(input_shape, conv_func):
    # generate dummy input sample and forward to get shape after conv layers
    batch_size = 1
    input = torch.rand(batch_size, *input_shape)
    output_feat = conv_func(input)
    conv_out_size = output_feat.data.view(batch_size, -1).size(1)
    return conv_out_size
# -------------------------------------------------------------------------------------------
#  ConvNet
# -------------------------------------------------------------------------------- -----------
class ConvNet3(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(ConvNet3, self).__init__()
        self.layers_names = ('conv1', 'conv2', 'FC1', 'FC_out')
        color_channels = input_shape[0]
        n_filt1 = 10
        n_filt2 = 20
        n_hidden_fc1 = 50
        self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=5)
        self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=5)
        conv_feat_size = get_size_of_conv_output(input_shape, self._forward_features)
        self.fc1 = linear_layer(conv_feat_size, n_hidden_fc1)
        self.fc_out = linear_layer(n_hidden_fc1, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def _forward_features(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), 2))
        x = F.elu(F.max_pool2d(self.conv2(x), 2))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_out(x)
        return x


@register_model
def Layer13s(num_classes=10, in_chans=3, base_width=8, **_):
    return layer13s(num_classes=num_classes, in_chans=in_chans, base_width=base_width)


# -------------------------------------------------------------------------------------------
#  OmConvNet
# -------------------------------------------------------------------------------- -----------
class OmConvNet_NoBN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(OmConvNet_NoBN, self).__init__()
        self.model_name = 'OmConv'
        n_in_channels = input_shape[0]
        n_filt1 = 64
        n_filt2 = 64
        n_filt3 = 64
        self.conv_layers = nn.Sequential(OrderedDict([
                ('conv1',  nn.Conv2d(n_in_channels, n_filt1, kernel_size=3)),
                ('relu1',  nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv2', nn.Conv2d(n_filt1, n_filt2, kernel_size=3)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv3', nn.Conv2d(n_filt2, n_filt3, kernel_size=3)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
                 ]))
        conv_out_size = get_size_of_conv_output(input_shape, self._forward_conv_layers)
        self.add_module('fc_out', nn.Linear(conv_out_size, output_dim))

        # Initialize weights
        #self._init_weights()

    def _forward_conv_layers(self, x, weights=None):
        if weights is None:
            x = self.conv_layers(x)
        else:
            x = F.conv2d(x, weights['conv_layers.conv1.weight'], weights['conv_layers.conv1.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.conv2d(x, weights['conv_layers.conv2.weight'], weights['conv_layers.conv2.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.conv2d(x, weights['conv_layers.conv3.weight'], weights['conv_layers.conv3.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x, weights=None):
        x = self._forward_conv_layers(x, weights)
        x = x.view(x.size(0), -1)
        if weights is None:
            x = self.fc_out(x)
        else:
            x = F.linear(x, weights['fc_out.weight'], weights['fc_out.bias'])
        return x


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_dim=10, out_dim=1):

        super(MLPModel, self).__init__()

        #self.fc0 = nn.Linear(input_dim, 256) 
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 32)    
        self.output = nn.Linear(32, out_dim)


    def forward(self, x):

        #x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x


#-------------------------NTK Models-------------------------------
class FCNTK(nn.Module):
    """
    Fully-connected NTK-style network (infinite-width approximation)
    """
    def __init__(self, in_dim=784, hidden_dim=8192, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class CNTK(nn.Module):
    """
    Convolutional NTK-style network (approx. infinite channels)
    """
    def __init__(self, in_chans=3, base_width=512, num_classes=10):
        super().__init__()
        k = base_width
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, k, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k, 2*k, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(2*k, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
