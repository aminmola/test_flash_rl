import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(CNN, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.value_stream = nn.Linear(256, 1)  
        self.advantage_stream = nn.Linear(256, action_size)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())
        return q_values