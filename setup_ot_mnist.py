import torch.nn as nn
import torch.nn.functional as F

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
    
class Encoder(nn.Module):
    def __init__(self, dim_h,n_z):
        super(Encoder, self).__init__()
        """
        The encoder structure in OT_Classifier
        """
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Conv2d(1, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
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

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        """
        The baseline model for MNIST
        """
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def rank(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x    
    

class ECLA(nn.Module):
    def __init__(self, dim_h,n_z):
        super(ECLA, self).__init__()
        """
        The encoder+classifier structure for MNIST.
        Train ECLA to compare with OT_Classifier.
        """
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Conv2d(1, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

        self.main2 = nn.Sequential(
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
        x = x.squeeze()
        x = self.fc(x)
        x = self.main2(x)
        return x