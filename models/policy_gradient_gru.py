import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from config import hyperparams
from device import device
from models.agent import Agent
from online_normalization import OnlineNormalization


# GRU-based Policy Network for Policy Gradient
class PolicyNetwork_GRU(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork_GRU, self).__init__()

        hidden_size = hyperparams['hidden_layer_size']
        gru_hidden_size = hyperparams['lstm_hidden_size']  # Using the GRU hidden size from hyperparameters
        num_layers = hyperparams['lstm_num_layers']

        self.fc1 = nn.Linear(state_dim, hidden_size)  # Initial fully connected layer
        self.gru = nn.GRU(hidden_size, gru_hidden_size, num_layers, batch_first=True)  # GRU layer
        self.fc2 = nn.Linear(gru_hidden_size, action_dim)  # Output layer for action probabilities
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, hidden_state=None):
        x = torch.relu(self.fc1(x))  # Pass through the first FC layer
        x = x.unsqueeze(1)  # Add a sequence dimension for the GRU

        if hidden_state is None:
            gru_out, hidden_state = self.gru(x)
        else:
            gru_out, hidden_state = self.gru(x, hidden_state)

        gru_out = gru_out[:, -1, :]  # Take the output from the last time step
        action_probs = torch.softmax(self.fc2(gru_out), dim=-1)  # Output action probabilities
        return action_probs, hidden_state  # Output the action probabilities and hidden state


# GRU-based Policy Gradient Agent (REINFORCE with GRU)
class PolicyGradientAgent_GRU(Agent):
    def __init__(self, train_env, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']

        state_dim = np.prod(state_shape)

        self.model = PolicyNetwork_GRU(state_dim, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = []

        self.train_env = train_env

        # Initialize OnlineNormalization for normalizing states
        self.normalizer = OnlineNormalization(state_dim)

    def choose_action(self, state, hidden_state=None):
        # Update and normalize the state before feeding into the model
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        action_probs, hidden_state = self.model(state_tensor, hidden_state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()
        return action, hidden_state

    def remember(self, state, action, reward):
        # Store the normalized state, action, and reward for the episode
        self.memory.append((state, action, reward))

    def compute_returns(self, rewards):
        """
        Compute the discounted returns for each time step in an episode.
        """
        discounted_returns = []
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return = reward + self.discount_factor * cumulative_return
            discounted_returns.insert(0, cumulative_return)
        return discounted_returns

    def learn(self):
        states, actions, rewards = zip(*self.memory)

        # Convert the list of numpy arrays to a single numpy array
        states = np.array(states)  # Efficient conversion

        # Compute discounted returns for the episode
        returns = self.compute_returns(rewards)

        # Normalize returns for stable training
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Convert states and actions to torch tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)

        # Compute loss as the negative log-probability of actions weighted by returns
        loss = 0
        hidden_state = None
        for i in range(len(states)):
            action_probs, hidden_state = self.model(states[i].unsqueeze(0), hidden_state)
            action_distribution = torch.distributions.Categorical(action_probs)
            log_prob = action_distribution.log_prob(actions[i])
            loss += -log_prob * returns[i]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after updating
        self.memory = []

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

            hidden_state = None  # Initialize hidden state for the GRU

            while not done:
                action, hidden_state = self.choose_action(state, hidden_state)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                next_state = next_state.flatten()

                # Store the transition in memory
                self.remember(state, action, reward)

                # Track the action and portfolio value
                episode_portfolio_values.append(info['portfolio_valuation'])  # Append portfolio value
                episode_actions.append(hyperparams['positions'][action])  # Track the action taken

                state = next_state
                total_reward += reward

            # Learn from the episode after it ends
            self.learn()

            if episode == n_episodes - 1:  # If it's the last episode, store the values and actions for plotting
                last_episode_portfolio_values = episode_portfolio_values
                last_episode_actions = episode_actions

            if episode % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

        return last_episode_portfolio_values, last_episode_actions  # Return values and actions from the last episode

    def get_action_probabilities(self, state, hidden_state=None):
        """
        Return action probabilities for a given state.
        """
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        # Convert the state to a tensor and move to the appropriate device (CPU/GPU)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)

        # Ensure no gradients are calculated during inference
        with torch.no_grad():
            action_probs, hidden_state = self.model(state_tensor, hidden_state)

        # Return the action probabilities as a numpy array
        return action_probs.cpu().numpy().flatten(), hidden_state
