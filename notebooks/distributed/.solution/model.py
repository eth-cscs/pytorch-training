import torch
import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):

         # Input shape
        assert x.shape[1:] == (3, 32, 32)

        # Apply first convolution and activation
        x = F.relu(self.conv1(x))

        # First convolution goes from 3 to 16 channels
        # With padding=1 and kernel_size=3 image size is preserved
        assert x.shape[1:] == (16, 32, 32)

        # Apply pooling
        x = self.pool(x)

        assert x.shape[1:] == (16, 16, 16)

        # Apply second convolution and activation
        x = F.relu(self.conv2(x))

        # First convolution goes from 16 to 32 channels
        # With padding=1 and kernel_size=3 image size is preserved
        assert x.shape[1:] == (32, 16, 16)

        # Apply pooling
        x = self.pool(x)

        assert x.shape[1:] == (32, 8, 8)

        # Apply third convolution and activation
        x = F.relu(self.conv3(x))

        # First convolution goes from 32 to 64 channels
        # With padding=1 and kernel_size=3 image size is preserved
        assert x.shape[1:] == (64, 8, 8)

        # Apply pooling
        x = self.pool(x) 

        assert x.shape[1:] == (64, 4, 4)

        # Flatten features for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten keeping batch size dynamic

        assert x.shape[1] == 64 * 4 * 4, f"Unexpected shape after flatten: {x.shape}"

        # Apply first fully connected layer and activation
        x = F.relu(self.fc1(x))

        assert x.shape[1:] == (512,), f"Expected (512,), got {x.shape[1:]}"

        # Apply second fully connected layer and log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)

        assert x.shape[1:] == (10,)

        return x
