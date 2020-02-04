import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_z, dim_h):
        super(Encoder, self).__init__()
        """
        The encoder structure in OT_Classifier
        """
        self.dim_h = dim_h
        self.n_z = n_z
        self.main1 = nn.Sequential( # 3 * 96 * 96
            nn.Conv2d(3, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True), # 48 * 48 
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True), # 24 * 24
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True), # 12 * 12
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True), # 6 * 6
            nn.Conv2d(self.dim_h * 8, self.dim_h * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 16),
            nn.ReLU(True), # 3 * 3
            nn.Conv2d(self.dim_h * 16, self.dim_h * 16, 3, 2, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 16),
            nn.ReLU(True), # 1 * 1
        )
        
        self.fc1 = nn.Linear(self.dim_h * (2 ** 4), self.n_z)
        
    def forward(self, x):
        x = self.main1(x)
        x = x.squeeze()
        x = self.fc1(x)
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
    
class SmallCNN(nn.Module):
    def __init__(self, n_z):
        super(SmallCNN, self).__init__()
        """
        The classifier structure in OT_Classifier
        """
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Linear(500, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.main(x)
        return x
