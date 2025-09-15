import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque
from RL.DuellingDDQN import DuellingDDQNWithAutoencoder, PrioritizedReplayBuffer

class EnhancedDQL:
    """Enhanced DQL agent with Duelling DDQN and Autoencoder"""

    def __init__(self, state_size, action_size, batch_size, learning_rate=0.001,
                 gamma=0.99, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.1,
                 update_rate=10, encoding_dim=128, use_prioritized_replay=True):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_rate = update_rate
        self.encoding_dim = encoding_dim
        self.use_prioritized_replay = use_prioritized_replay

        # Initialize networks
        self.main_network = DuellingDDQNWithAutoencoder(
            state_size, action_size, encoding_dim
        )
        self.target_network = DuellingDDQNWithAutoencoder(
            state_size, action_size, encoding_dim
        )

        # Copy main network weights to target network
        self.target_network.load_state_dict(
            copy.deepcopy(self.main_network.state_dict())
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.main_network.parameters(), lr=learning_rate
        )

        # Experience replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        else:
            self.replay_buffer = deque(maxlen=10000)

        # Loss function
        self.loss_func = nn.MSELoss()

        # Training metrics
        self.training_step = 0
        self.losses = []

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if self.use_prioritized_replay:
            self.replay_buffer.add(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))

    def multiaction_selection(self, state, C, comm_round):
        """Enhanced multi-action selection using Duelling DDQN"""
        m = int(max(C * self.action_size, 1))

        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state.unsqueeze(0))

        # Get top-C actions based on Q-values
        top_actions = torch.topk(q_values[0], m).indices.tolist()

        # Epsilon-greedy selection with exploration
        selected_clients = []
        remaining_actions = list(range(self.action_size))

        for _ in range(m):
            if random.random() < self.epsilon and remaining_actions:
                # Explore: random selection from remaining actions
                action = random.choice(remaining_actions)
            elif top_actions:
                # Exploit: select from top actions
                action = top_actions.pop(0)
            else:
                # Fallback: random selection
                action = random.choice(remaining_actions) if remaining_actions else 0

            if action in remaining_actions:
                selected_clients.append(action)
                remaining_actions.remove(action)
                if action in top_actions:
                    top_actions.remove(action)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return sorted(selected_clients)

    def train(self, comm_round):
        """Enhanced training with Duelling DDQN"""
        if self.use_prioritized_replay:
            if len(self.replay_buffer) < self.batch_size:
                return 0.0

            # Sample from prioritized replay buffer
            minibatch, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights)
        else:
            if len(self.replay_buffer) < self.batch_size:
                minibatch = list(self.replay_buffer)
            else:
                minibatch = random.sample(self.replay_buffer, self.batch_size)
            weights = torch.ones(len(minibatch))

        # Prepare batch data
        states = torch.stack([sample[0] for sample in minibatch])
        actions = torch.tensor([sample[1] for sample in minibatch])
        rewards = torch.tensor([sample[2] for sample in minibatch], dtype=torch.float32)
        next_states = torch.stack([sample[3] for sample in minibatch])
        dones = torch.tensor([sample[4] for sample in minibatch], dtype=torch.bool)

        # Current Q-values
        self.main_network.train()
        current_q_values = self.main_network(states)

        # Handle multi-action case
        if len(actions.shape) > 1:
            # Multi-action scenario
            current_q_selected = current_q_values.gather(1, actions)
        else:
            # Single action scenario
            current_q_selected = current_q_values.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            self.target_network.eval()
            self.main_network.eval()

            # Use main network to select actions
            next_q_main = self.main_network(next_states)
            next_actions = next_q_main.argmax(dim=1)

            # Use target network to evaluate selected actions
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Set Q-values for terminal states to 0
            next_q_values[dones] = 0

            # Compute target Q-values
            if len(rewards.shape) > 1:
                target_q_values = rewards.mean(dim=1) + self.gamma * next_q_values
            else:
                target_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        if len(current_q_selected.shape) > 1:
            loss = (weights.unsqueeze(1) *
                   self.loss_func(current_q_selected, target_q_values.unsqueeze(1))).mean()
        else:
            loss = (weights *
                   F.mse_loss(current_q_selected, target_q_values, reduction='none')).mean()

        # Update priorities for prioritized replay
        if self.use_prioritized_replay:
            with torch.no_grad():
                td_errors = torch.abs(current_q_selected.squeeze() - target_q_values).detach().numpy()
                priorities = td_errors + 1e-6  # Small epsilon to avoid zero priorities
                self.replay_buffer.update_priorities(indices, priorities)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(
            copy.deepcopy(self.main_network.state_dict())
        )

    def pretrain_autoencoder(self, state_data, epochs=50):
        """Pretrain autoencoder on state representations"""
        print("Pretraining autoencoder...")

        # Create dataloader from state data
        dataset = torch.utils.data.TensorDataset(torch.stack(state_data))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Pretrain autoencoder
        self.main_network.pretrain_autoencoder(dataloader, epochs)

        # Update target network
        self.target_network.load_state_dict(
            copy.deepcopy(self.main_network.state_dict())
        )

        print("Autoencoder pretraining completed")

    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)

    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']

    def get_training_metrics(self):
        """Get training metrics for analysis"""
        return {
            'losses': self.losses,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }