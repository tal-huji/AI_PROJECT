import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from config import hyperparams
from device import device
from models.agent import Agent
from online_normalization import OnlineNormalization

# CNN + GRU-based DQN model without batching
class DQN_GRU_CNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_GRU_CNN, self).__init__()

        hidden_size = hyperparams['hidden_layer_size']
        gru_hidden_size = hyperparams['lstm_hidden_size']  # Reusing the same hyperparameter for GRU
        num_layers = hyperparams['lstm_num_layers']

        # CNN layers to extract features from raw state input
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # After CNN, flatten output and feed into fully connected layers
        self.cnn_output_dim = (state_dim // 2) * 32  # Assuming pooling reduces the dimension by half
        self.fc1 = nn.Linear(self.cnn_output_dim, hidden_size)

        # Additional fully connected layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Additional layer 1
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional layer 2

        # GRU layer
        self.gru = nn.GRU(hidden_size, gru_hidden_size, num_layers, batch_first=True)

        # Output layer for Q-values
        self.fc_output = nn.Linear(gru_hidden_size, action_dim)

    def forward(self, x, hidden_state=None):
        # x shape: (sequence_length, state_dim)
        seq_len, state_dim = x.size()

        # Reshape input for CNN: (sequence_length, 1, state_dim)
        x = x.unsqueeze(1)

        # Pass through CNN layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # Apply max pooling

        # Flatten the output for the fully connected layer
        x = x.view(seq_len, -1)
        x = torch.relu(self.fc1(x))  # Pass through the first fully connected layer

        # Pass through the additional fully connected layers
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Reshape for GRU: (1, sequence_length, hidden_size)
        x = x.unsqueeze(0)

        # Pass through the GRU layer
        if hidden_state is None:
            gru_out, hidden_state = self.gru(x)
        else:
            gru_out, hidden_state = self.gru(x, hidden_state)

        # Output Q-values for each time step
        q_values = self.fc_output(gru_out.squeeze(0))  # Output shape: (sequence_length, action_dim)
        return q_values, hidden_state

# Replay Buffer that stores sequences
class ReplayBuffer:
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = []
        self.position = 0

    def add(self, episode):
        """Add an episode (sequence of transitions) to the buffer."""
        if len(episode) >= self.sequence_length:
            for i in range(len(episode) - self.sequence_length + 1):
                sequence = episode[i:i + self.sequence_length]
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                self.buffer[self.position] = sequence
                self.position = (self.position + 1) % self.capacity

    def sample(self):
        """Sample a single sequence from the buffer."""
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)

# DQN Agent with CNN + GRU without batching
class DQNAgent_GRU_CNN(Agent):
    def __init__(self, train_env, state_shape, action_size):
        self.state_shape = state_shape  # Should be (sequence_length, state_dim)
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_min = hyperparams['exploration_min']
        self.exploration_decay = hyperparams['exploration_decay']

        state_dim = state_shape[-1]  # State dimension
        self.sequence_length = hyperparams['sequence_length']

        # Initialize the CNN-GRU model
        self.model = DQN_GRU_CNN(state_dim, action_size).to(device)
        self.target_model = DQN_GRU_CNN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Replay Buffer that stores sequences
        self.memory = ReplayBuffer(hyperparams['memory_size'], self.sequence_length)
        self.train_env = train_env

        # Initialize OnlineNormalization for normalizing states
        self.normalizer = OnlineNormalization(state_dim)

    def choose_action(self, state, hidden_state=None):
        # Update and normalize the state before feeding into the model
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        if np.random.rand() <= self.exploration_rate:
            return np.random.randint(self.action_size), hidden_state

        # Prepare the state tensor
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, state_dim)
        with torch.no_grad():
            q_values, hidden_state = self.model(state_tensor.squeeze(0), hidden_state)
        return q_values.argmax().item(), hidden_state

    def remember(self, episode_transitions):
        # Update and normalize states in the episode
        normalized_episode = []
        for state, action, reward, next_state, done in episode_transitions:
            self.normalizer.update(state)
            self.normalizer.update(next_state)

            normalized_state = self.normalizer.normalize(state)
            normalized_next_state = self.normalizer.normalize(next_state)

            normalized_episode.append((normalized_state, action, reward, normalized_next_state, done))
        # Store the normalized episode in the replay buffer
        self.memory.add(normalized_episode)

    def learn(self):
        if len(self.memory) == 0:
            return

        # Sample a single sequence from the replay buffer
        sequence = self.memory.sample()

        states, actions, rewards, next_states, dones = zip(*sequence)

        # Convert lists to tensors
        state_sequence = torch.FloatTensor(np.stack(states)).to(device)  # Shape: (sequence_length, state_dim)
        action_sequence = torch.LongTensor(actions).to(device)  # Shape: (sequence_length,)
        reward_sequence = torch.FloatTensor(rewards).to(device)  # Shape: (sequence_length,)
        next_state_sequence = torch.FloatTensor(np.stack(next_states)).to(device)
        done_sequence = torch.FloatTensor(dones).to(device)

        # Initialize hidden state
        hidden_state = None

        # Compute current Q values for all time steps
        current_q_values, _ = self.model(state_sequence, hidden_state)  # Output shape: (sequence_length, action_dim)
        # Select the Q-values for the actions taken at each time step
        current_q_values = current_q_values.gather(1, action_sequence.unsqueeze(1)).squeeze(1)

        # Compute next Q values using the target network only
        with torch.no_grad():
            next_q_values_target, _ = self.target_model(next_state_sequence, hidden_state)
            next_q_values = next_q_values_target.max(1)[0]  # Select the max Q-value from the target model

        # Compute target Q values
        target_q_values = reward_sequence + (1 - done_sequence) * self.discount_factor * next_q_values

        # Compute loss over all time steps
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

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

            hidden_state = None  # Initialize hidden state for the GRU

            self.model.train()

            episode_transitions = []  # Store transitions for the episode

            while not done:
                # Choose action using the current policy
                action, hidden_state = self.choose_action(state, hidden_state)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                next_state = next_state.flatten()

                # Store transition
                episode_transitions.append((state, action, reward, next_state, done))

                # Track the action and portfolio value
                episode_portfolio_values.append(info['portfolio_valuation'])  # Append portfolio value
                episode_actions.append(hyperparams['positions'][action])  # Track the action taken

                state = next_state
                total_reward += reward

                # Learn from the replay buffer after each step
                self.learn()

            # At the end of the episode, store the episode transitions in the replay buffer
            self.remember(episode_transitions)

            if episode == n_episodes - 1:  # If it's the last episode, store the values and actions for plotting
                last_episode_portfolio_values = episode_portfolio_values
                last_episode_actions = episode_actions

            # Update the target model periodically
            if episode % 10 == 0:
                self.update_target_model()
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

        return last_episode_portfolio_values, last_episode_actions  # Return values and actions from the last episode