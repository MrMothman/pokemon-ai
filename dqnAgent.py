# dqn_agent.py
import torch
import dqnModel as DQNModel
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    # Implement a custom replay buffer
    pass

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000):
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = 0.95

    def select_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.model.output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.model(torch.Tensor(state))
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.Tensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        dones = torch.Tensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1).long())
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
