import torch.nn as nn

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

class Encoder(nn.Module):
    def __init__(self, n_z):
        super(Encoder, self).__init__()
        """
        The encoder structure in OT_Classifier
        """        
        self.feature = make_layers(cfg['E'])
        
        self.fc1 = nn.Linear(512, n_z)
        
    def forward(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225    
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x    

class SmallCNN(nn.Module):
    def __init__(self, n_z):
        super(SmallCNN, self).__init__()
        """
        The classifier structure in OT_Classifier
        """
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, 512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim_h, n_z):
        super(Discriminator, self).__init__()
        """
        The discriminator structure in OT_Classifier
        """
        self.dim_h = dim_h
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.LogSigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x   
    