import torch.nn as nn
import torch

class MLP(nn.Module):
    # RL module
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        
        x = self.fc1(state)
        x = self.relu(x)

        # we dont want to activate it, we need the row numbers
        Q_values = self.fc2(x)
        
        return Q_values
class DuelingMLP(nn.Module):
    """Dueling network: separate value and advantage streams.
    Output Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Advantage stream
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state):
        x = self.feature(state)
        V = self.value(x)              # [batch, 1]
        A = self.adv(x)                # [batch, actions]
        A_centered = A - A.mean(dim=1, keepdim=True)
        Q = V + A_centered
        return Q
