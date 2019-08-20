from __future__ import print_function
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from imports import *



class AlexNet(nn.Module):
    """
        AlexNet architecture responsible of extracting image features
        from FC7 after ReLU

        output: 4096-d feature vector
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def CNN(pretrained=False, **kwargs):
    model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    alex_net = AlexNet(**kwargs)
    if pretrained:
        # grab the weights of the layers
        saved_model = model_zoo.load_url(model_url)
        # removes the weights of the last layer since we dont need it
        del saved_model["classifier.6.weight"]
        del saved_model["classifier.6.bias"]
        alex_net.load_state_dict(saved_model)
    return alex_net


class CnnRnn(nn.Module):
    def __init__(self, pre_trained=True, input_size=4096, hidden_size=64,
                 num_layers=1, bias=True, batch_first=True,
                 dropout=0, bidirectional=False,global_dict={}):
        """
            Initializes the CNN-RNN structure for regression
            :param pre_trained: If True, returns a model pre-trained on ImageNet
            :param input_size: The number of expected features in the input `x`
            :param hidden_size: The number of features in the hidden state `h`
            :param num_layers: Number of recurrent layers.
            :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            :param batch_first: If ``True``, then the input and output tensors are provided
                as `(batch, seq, feature)`. Default: ``False``
            :param dropout: The dropout probability.
            :param bidirectional: If ``True``, becomes a bidirectional RNN.
            :param frame_level: If ``True``, the RNN network is Many to Many, else Many to One
        """
        super(CnnRnn, self).__init__()
        self.cnn = CNN(pre_trained)
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first,
                          dropout, bidirectional)
      
        self.linear = nn.Linear(hidden_size,global_dict['labels_dict']['number'])

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, h_n = self.rnn(r_in)
     
        return self.linear(r_out[:, -1, :])
