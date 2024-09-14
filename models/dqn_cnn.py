import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from config import hyperparams
from device import device
from models.agent import Agent
from online_normalization import OnlineNormalization

# CNN-based DQN model without GRU
class DQN_CNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_CNN, self).__init__()

        hidden_size = hyperparams['hidden_layer_size']

        # CNN layers to extract features from raw state input
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # After CNN, flatten output and feed into fully connected layers
        cnn_output_dim = state_dim // 2  # Assuming pooling reduces the dimension by half
        self.fc1 = nn.Linear(cnn_output_dim * 32, hidden_size)

        # Additional fully connected layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Additional layer 1
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional layer 2

        # Output layer for Q-values
        self.fc_output = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        # Reshape input for CNN: (batch_size, 1, state_dim)
        x = x.unsqueeze(1)

        # Pass through CNN layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # Apply max pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = torch.relu(self.fc2(x))  # Second fully connected layer
        x = torch.relu(self.fc3(x))  # Third fully connected layer

        # Output Q-values
        return self.fc_output(x)

# Replay Buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        # Sample a single transition
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)

# DQN Agent with CNN only
class DQNAgent_CNN(Agent):
    def __init__(self, train_env, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_min = hyperparams['exploration_min']
        self.exploration_decay = hyperparams['exploration_decay']

        state_dim = np.prod(state_shape)

        # Initialize the CNN model
        self.model = DQN_CNN(state_dim, action_size).to(device)
        self.target_model = DQN_CNN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Replay Buffer
        self.memory = ReplayBuffer(hyperparams['memory_size'])
        self.train_env = train_env

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

        # Store the transition in the replay buffer
        self.memory.add(normalized_state, action, reward, normalized_next_state, done)

    def learn(self):
        if len(self.memory) == 0:
            return

        # Sample a single transition from the replay buffer
        state, action, reward, next_state, done = self.memory.sample()

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        done = torch.FloatTensor([done]).to(device)

        # Compute Q(s, a)
        current_q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Regular DQN: Use the target model to both select and evaluate the best action in the next state
        next_q_values_target = self.target_model(next_state)
        next_q_value = next_q_values_target.max(1)[0]  # Max Q-value for the next state

        # Compute the target Q-value
        target_q_value = reward + (1 - done) * self.discount_factor * next_q_value

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_value.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Decay exploration rate
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_agent(self):
        super().train_agent()

        last_episode_portfolio_values = []  # Track values only for the last episode
        last_episode_actions = []  # Track actions taken in the last episode
        n_episodes = hyperparams['n_episodes']

        for episode in range(n_episodes):
            state, _ = self.train_env.reset()
            state = state.flatten()
            total_reward = 0
            done = False
            episode_portfolio_values = []  # Track portfolio values for this episode
            episode_actions = []  # Track actions taken in this episode

            self.model.train()

            while not done:
                # Choose action using the current policy
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                next_state = next_state.flatten()

                # Store transition in the replay memory
                self.remember(state, action, reward, next_state, done)

                # Learn from the replay buffer after each step
                self.learn()

                # Track the action and portfolio value
                episode_portfolio_values.append(info['portfolio_valuation'])  # Append portfolio value
                episode_actions.append(hyperparams['positions'][action])  # Track the action taken

                state = next_state
                total_reward += reward

            if episode == n_episodes - 1:  # If it's the last episode, store the values and actions for plotting
                last_episode_portfolio_values = episode_portfolio_values
                last_episode_actions = episode_actions

            # Update the target model periodically
            if episode % 10 == 0:
                self.update_target_model()
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

        return last_episode_portfolio_values, last_episode_actions  # Return values and actions from the last episode
