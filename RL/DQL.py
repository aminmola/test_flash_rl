
import copy
import random
from collections import deque
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling architecture for Q-value estimation."""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        a = self.advantage(f)
        v = self.value(f)
        # Combine streams: Q(s,a) = V(s) + A(s) - mean(A(s))
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


class DQL:
    """Double DQN with Dueling network (drop-in replacement for original DQL class)."""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        memory_size: int = 50_000,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        update_rate: int = 10,
        device: str | None = None,
        hidden_sizes: Tuple[int, int] = (256, 256),
        seed: int = 42,
        flag: bool = False,  # keep signature compatibility
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_rate = update_rate
        self.learn_step = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Main & target networks (dueling)
        self.main_network = DuelingQNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.update_target_network(hard=True)

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    # ----------------- Interaction -----------------
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    # ----------------- Learning -----------------
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates
        q_pred = self.main_network(states_t).gather(1, actions_t)

        # Double DQN targets:
        # next action from main network (argmax)
        with torch.no_grad():
            next_q_main = self.main_network(next_states_t)
            next_actions = torch.argmax(next_q_main, dim=1, keepdim=True)
            # evaluate those actions with target network
            next_q_target = self.target_network(next_states_t).gather(1, next_actions)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        loss = self.criterion(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_rate == 0:
            self.update_target_network(hard=False)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_network(self, hard: bool = False, tau: float = 0.01):
        if hard:
            self.target_network.load_state_dict(copy.deepcopy(self.main_network.state_dict()))
        else:
            # soft update
            for tgt, src in zip(self.target_network.parameters(), self.main_network.parameters()):
                tgt.data.copy_(tgt.data * (1.0 - tau) + src.data * tau)
