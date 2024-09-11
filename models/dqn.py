# models/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from config import hyperparams
from device import device
from models.agent import Agent
from online_normalization import OnlineNormalization


# DQN model with Layer Normalization (for small batches)
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_size)
#         self.ln1 = nn.LayerNorm(hidden_size)  # Layer Norm instead of Batch Norm
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.ln2 = nn.LayerNorm(hidden_size)  # Layer Norm instead of Batch Norm
#         self.fc3 = nn.Linear(hidden_size, action_dim)
#         self.init_weights()
#
#     def init_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 torch.nn.init.zeros_(module.bias)
#
#     def forward(self, x):
#         x = torch.relu(self.ln1(self.fc1(x)))  # LayerNorm after first linear layer
#         x = torch.relu(self.ln2(self.fc2(x)))  # LayerNorm after second linear layer
#         return self.fc3(x)  # No LayerNorm after output layer


# DQN model without normalization
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN Agent
class DQNAgent(Agent):
    def __init__(self, train_env, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_min = hyperparams['exploration_min']
        self.exploration_decay = hyperparams['exploration_decay']

        state_dim = np.prod(state_shape)
        self.model = DQN(state_dim, action_size, hyperparams['hidden_layer_size']).to(device)
        self.target_model = DQN(state_dim, action_size, hyperparams['hidden_layer_size']).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = []

        self.train_env = train_env
        # self.test_env = test_env

        # Initialize OnlineNormalization for normalizing states
        self.normalizer = OnlineNormalization(state_dim)

    def choose_action(self, state):
        # Update and normalize the state before feeding into the model
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        if np.random.rand() <= self.exploration_rate:
            return np.random.randint(self.action_size)

        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        # Update and normalize both state and next_state before storing
        self.normalizer.update(state)
        self.normalizer.update(next_state)

        normalized_state = self.normalizer.normalize(state)
        normalized_next_state = self.normalizer.normalize(next_state)

        self.memory.append((normalized_state, action, reward, normalized_next_state, done))
        if len(self.memory) > hyperparams['memory_size']:
            self.memory.pop(0)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Train agent
    # Train agent
    def train_agent(self):
        super().train_agent()

        last_episode_portfolio_values = []  # Track values only for the last episode
        last_episode_actions = []  # Track actions taken in the last episode
        n_episodes = hyperparams['n_episodes']

        for episode in range(n_episodes):
            state, _ = self.train_env.reset(seed=hyperparams['seed'])
            state = state.flatten()
            total_reward = 0
            done = False
            episode_portfolio_values = []  # Track portfolio values for this episode
            episode_actions = []  # Track actions taken in this episode

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                next_state = next_state.flatten()

                self.remember(state, action, reward, next_state, done)
                self.learn(hyperparams['batch_size'])

                # Track the action and portfolio value
                episode_portfolio_values.append(info['portfolio_valuation'])  # Append portfolio value
                episode_actions.append(hyperparams['positions'][action])  # Track the action taken

                state = next_state
                total_reward += reward

            if episode == n_episodes - 1:  # If it's the last episode, store the values and actions for plotting
                last_episode_portfolio_values = episode_portfolio_values
                last_episode_actions = episode_actions

            if episode % 10 == 0:
                self.update_target_model()
                print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

        return last_episode_portfolio_values, last_episode_actions  # Return values and actions from the last episode

    # Test agent
    def test_agent(self):
        pass

    def get_q_values(self, state):
        """
        Return Q-values for all actions based on the given state.
        """
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        # Convert the state to a tensor and move to the appropriate device (CPU/GPU)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)

        # Ensure no gradients are calculated during inference
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Return the Q-values as a numpy array
        return q_values.cpu().numpy().flatten()
