import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction and feature extraction"""

    def __init__(self, input_dim, encoding_dim=128):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class DuellingNetwork(nn.Module):
    """Duelling DQN architecture with separate value and advantage streams"""

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DuellingNetwork, self).__init__()

        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)

        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

class DuellingDDQNWithAutoencoder(nn.Module):
    """Combined Duelling DDQN with Autoencoder for enhanced node selection"""

    def __init__(self, state_dim, action_dim, encoding_dim=128, hidden_dim=512):
        super(DuellingDDQNWithAutoencoder, self).__init__()

        self.autoencoder = Autoencoder(state_dim, encoding_dim)
        self.duelling_network = DuellingNetwork(encoding_dim, action_dim, hidden_dim)

        # Freeze autoencoder during RL training (optional)
        self.freeze_autoencoder = False

    def encode_state(self, state):
        """Encode state using autoencoder"""
        encoded, _ = self.autoencoder(state)
        return encoded

    def forward(self, state):
        # Encode state first
        if self.freeze_autoencoder:
            with torch.no_grad():
                encoded_state = self.encode_state(state)
        else:
            encoded_state = self.encode_state(state)

        # Get Q-values from duelling network
        q_values = self.duelling_network(encoded_state)
        return q_values

    def pretrain_autoencoder(self, dataloader, epochs=50, lr=0.001):
        """Pretrain the autoencoder on state representations"""
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                encoded, decoded = self.autoencoder(batch)
                loss = criterion(decoded, batch)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Autoencoder Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def set_freeze_autoencoder(self, freeze=True):
        """Freeze/unfreeze autoencoder parameters"""
        self.freeze_autoencoder = freeze
        for param in self.autoencoder.parameters():
            param.requires_grad = not freeze

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for enhanced learning"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)