import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
import random
from collections import deque
import copy

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extraction
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

    def forward(self, state):
        shared_features = self.shared_layer(state)
        policy = self.actor(shared_features)
        value = self.critic(shared_features)
        return policy, value

class PPOAgent:
    """Proximal Policy Optimization agent for client selection"""

    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99,
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Networks
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Memory
        self.memory = []

    def select_clients(self, state, C, deterministic=False):
        """Select clients using PPO policy"""
        m = int(max(C * self.action_dim, 1))

        with torch.no_grad():
            policy, value = self.actor_critic(state.unsqueeze(0))

        if deterministic:
            # Select top-m clients
            _, top_indices = torch.topk(policy[0], m)
            selected_clients = top_indices.tolist()
        else:
            # Sample from policy distribution
            dist = Categorical(policy[0])
            selected_clients = []
            remaining_actions = list(range(self.action_dim))

            for _ in range(m):
                if remaining_actions:
                    # Create masked distribution
                    mask = torch.zeros(self.action_dim)
                    mask[remaining_actions] = 1
                    masked_probs = policy[0] * mask
                    masked_probs = masked_probs / masked_probs.sum()

                    if masked_probs.sum() > 0:
                        masked_dist = Categorical(masked_probs)
                        action = masked_dist.sample().item()
                        selected_clients.append(action)
                        remaining_actions.remove(action)

        return sorted(selected_clients)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self, epochs=10):
        """Update policy using PPO"""
        if len(self.memory) == 0:
            return 0.0

        # Prepare batch data
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.tensor([m['action'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory], dtype=torch.float32)
        next_states = torch.stack([m['next_state'] for m in self.memory])
        dones = torch.tensor([m['done'] for m in self.memory], dtype=torch.float32)
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])

        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, next_states, dones)
        advantages = self._calculate_advantages(states, returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(epochs):
            # Forward pass
            policies, values = self.actor_critic(states)

            # Calculate new log probabilities
            new_log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)).squeeze())

            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy loss
            entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        # Clear memory
        self.memory = []
        return total_loss / epochs

    def _calculate_returns(self, rewards, next_states, dones):
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        return returns

    def _calculate_advantages(self, states, returns):
        """Calculate advantages using TD error"""
        with torch.no_grad():
            _, values = self.actor_critic(states)
            advantages = returns - values.squeeze()
        return advantages

class SACCritic(nn.Module):
    """Critic network for SAC"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

class SACActor(nn.Module):
    """Actor network for SAC with reparameterization trick"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim//2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim//2, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.randn_like(mean)

        # Reparameterization trick
        action = mean + std * normal
        log_prob = -0.5 * (normal.pow(2) + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Apply tanh squashing
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1, keepdim=True)

        return action, log_prob

class SACAgent:
    """Soft Actor-Critic agent for continuous client selection"""

    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.actor = SACActor(state_dim, action_dim)
        self.critic = SACCritic(state_dim, action_dim)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Experience replay
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def select_clients(self, state, C):
        """Select clients using SAC policy"""
        m = int(max(C * self.action_dim, 1))

        with torch.no_grad():
            action, _ = self.actor.sample(state.unsqueeze(0))
            action = action[0]

        # Convert continuous actions to discrete client selection
        action_probs = torch.softmax(action, dim=0)
        _, top_indices = torch.topk(action_probs, m)

        return sorted(top_indices.tolist())

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        """Update SAC networks"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Update alpha
        alpha_loss = self._update_alpha(states)

        # Soft update target networks
        self._soft_update()

        return critic_loss + actor_loss + alpha_loss

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states):
        """Update actor network"""
        actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states):
        """Update temperature parameter"""
        actions, log_probs = self.actor.sample(states)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class MARLAgent:
    """Multi-Agent Reinforcement Learning for collaborative client selection"""

    def __init__(self, state_dim, action_dim, num_agents=3, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Create multiple agents
        self.agents = []
        for i in range(num_agents):
            agent = {
                'actor_critic': ActorCritic(state_dim, action_dim),
                'optimizer': optim.Adam(ActorCritic(state_dim, action_dim).parameters(), lr=lr),
                'memory': []
            }
            self.agents.append(agent)

        # Shared experience buffer
        self.shared_memory = deque(maxlen=5000)

    def select_clients(self, state, C):
        """Collaborative client selection using multiple agents"""
        m = int(max(C * self.action_dim, 1))
        agent_selections = []

        # Each agent makes its selection
        for agent in self.agents:
            with torch.no_grad():
                policy, _ = agent['actor_critic'](state.unsqueeze(0))

            # Sample from policy
            dist = Categorical(policy[0])
            agent_probs = policy[0].detach().numpy()
            agent_selections.append(agent_probs)

        # Combine agent decisions using voting/averaging
        combined_probs = np.mean(agent_selections, axis=0)

        # Select top-m clients based on combined probabilities
        top_indices = np.argsort(combined_probs)[-m:]

        return sorted(top_indices.tolist())

    def store_shared_transition(self, state, action, reward, next_state, done):
        """Store transition in shared memory"""
        self.shared_memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def update_agents(self, epochs=5):
        """Update all agents using shared experience"""
        if len(self.shared_memory) < 64:
            return 0.0

        total_loss = 0

        for agent in self.agents:
            # Sample from shared memory
            batch = random.sample(self.shared_memory, min(64, len(self.shared_memory)))

            # Prepare batch data
            states = torch.stack([b['state'] for b in batch])
            actions = torch.tensor([b['action'] for b in batch])
            rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)
            next_states = torch.stack([b['next_state'] for b in batch])
            dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32)

            # Calculate returns
            returns = self._calculate_returns(rewards, dones)

            for _ in range(epochs):
                # Forward pass
                policies, values = agent['actor_critic'](states)

                # Calculate advantages
                advantages = returns - values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss (simplified)
                log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
                policy_loss = -(log_probs * advantages.detach()).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)

                # Entropy loss
                entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()
                entropy_loss = -0.01 * entropy

                # Total loss
                loss = policy_loss + 0.5 * value_loss + entropy_loss

                # Update
                agent['optimizer'].zero_grad()
                loss.backward()
                agent['optimizer'].step()

                total_loss += loss.item()

        return total_loss / (len(self.agents) * epochs)

    def _calculate_returns(self, rewards, dones, gamma=0.99):
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

class AdvancedRLSelector:
    """Unified interface for advanced RL algorithms"""

    def __init__(self, state_dim, action_dim, algorithm="PPO", **kwargs):
        self.algorithm = algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim

        if algorithm == "PPO":
            self.agent = PPOAgent(state_dim, action_dim, **kwargs)
        elif algorithm == "SAC":
            self.agent = SACAgent(state_dim, action_dim, **kwargs)
        elif algorithm == "MARL":
            self.agent = MARLAgent(state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def select_clients(self, state, C):
        """Select clients using the specified algorithm"""
        return self.agent.select_clients(state, C)

    def store_transition(self, *args, **kwargs):
        """Store transition in agent memory"""
        if hasattr(self.agent, 'store_transition'):
            return self.agent.store_transition(*args, **kwargs)
        elif hasattr(self.agent, 'store_shared_transition'):
            return self.agent.store_shared_transition(*args, **kwargs)

    def update(self, **kwargs):
        """Update agent"""
        if hasattr(self.agent, 'update'):
            return self.agent.update(**kwargs)
        elif hasattr(self.agent, 'update_agents'):
            return self.agent.update_agents(**kwargs)

    def save_model(self, filepath):
        """Save agent model"""
        if hasattr(self.agent, 'actor_critic'):
            torch.save(self.agent.actor_critic.state_dict(), f"{filepath}_{self.algorithm}.pth")
        elif hasattr(self.agent, 'actor'):
            torch.save({
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict()
            }, f"{filepath}_{self.algorithm}.pth")

    def load_model(self, filepath):
        """Load agent model"""
        if hasattr(self.agent, 'actor_critic'):
            self.agent.actor_critic.load_state_dict(torch.load(f"{filepath}_{self.algorithm}.pth"))
        elif hasattr(self.agent, 'actor'):
            checkpoint = torch.load(f"{filepath}_{self.algorithm}.pth")
            self.agent.actor.load_state_dict(checkpoint['actor'])
            self.agent.critic.load_state_dict(checkpoint['critic'])