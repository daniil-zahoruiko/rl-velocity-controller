import torch.nn as nn
import torch

class ActorNetwork(nn.Module):
    def __init__(self, num_thrusters):
        super(ActorNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(18 + num_thrusters, 600),
            nn.ReLU(),
            nn.Linear(600, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, num_thrusters),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)

class CriticNetwork(nn.Module):
    def __init__(self, num_thrusters):
        super(CriticNetwork, self).__init__()
        self.num_thrusters = num_thrusters
        self.linear = nn.Linear(18 + num_thrusters, 600)
        self.stack = nn.Sequential(
            nn.Linear(600 + num_thrusters, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear(x[:-self.num_thrusters])
        return self.stack(torch.cat((out, x[-self.num_thrusters:])))