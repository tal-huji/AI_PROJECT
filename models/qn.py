import numpy as np
import random
from collections import defaultdict
from config import dynamic_features_arr, hyperparams
from models.agent import Agent
from online_normalization import OnlineNormalization


class QLearningAgent(Agent):
    def __init__(self, train_env, test_env, action_size, min_max_array):
        self.train_env = train_env
        self.test_env = test_env
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_decay = hyperparams['exploration_decay']
        self.exploration_min = hyperparams['exploration_min']

        # Use the ordered min_max_array (array of tuples with min and max for each feature)
        self.min_max_array = min_max_array
        self.num_bins = hyperparams['num_bins']  # Use 20 bins as per your request

        # Initialize Q-table (discrete state, action pairs)
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def _discretize_state(self, state):
        """
        Convert continuous state to a discrete state using 20 bins for each feature.
        Flatten the state if it's multi-dimensional and use the ordered min_max_array.
        Use numpy.linspace() to ensure consistent binning.
        """
        if isinstance(state, np.ndarray):
            state = state.flatten()  # Flatten the array if it's multidimensional
        elif isinstance(state, list):
            state = np.array(state).flatten()  # Convert to numpy array and flatten

        discrete_state = []
        for i, value in enumerate(state):
            min_val, max_val = self.min_max_array[i]  # Use the i-th tuple from min_max_array

            # Handle cases where min_val and max_val are the same (no range)
            if max_val == min_val:
                bin_index = 0  # If no range, assign to the first bin
            else:
                # Generate bin edges using linspace between min_val and max_val with num_bins
                bin_edges = np.linspace(min_val, max_val, self.num_bins + 1)

                # Find the bin index where the value belongs
                bin_index = np.digitize(value, bin_edges) - 1

                # Ensure bin_index is within the valid range [0, num_bins - 1]
                bin_index = min(max(bin_index, 0), self.num_bins - 1)

            discrete_state.append(bin_index)

        # Convert the discrete state list to a tuple to use as a key in Q-table
        return tuple(discrete_state)

    def choose_action(self, state):
        """
        Choose an action based on the Q-table or exploration.
        """




        discrete_state = self._discretize_state(state)
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[discrete_state])


    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning algorithm.
        """
        discrete_state = self._discretize_state(state)
        next_discrete_state = self._discretize_state(next_state)

        current_q = self.q_table[discrete_state][action]

        if not done:
            best_future_q = np.max(self.q_table[next_discrete_state])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                    reward + self.discount_factor * best_future_q)
        else:
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * reward

        self.q_table[discrete_state][action] = new_q

    def train_agent(self):
        """
        Train the Q-learning agent by interacting with the environment.
        """
        last_episode_portfolio_values = []
        n_episodes = hyperparams['n_episodes']

        for episode in range(n_episodes):
            state, _ = self.train_env.reset(seed=42)
            total_reward = 0
            done = False
            episode_portfolio_values = []

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated

                self.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                episode_portfolio_values.append(info['portfolio_valuation'])

            if episode == n_episodes - 1:
                last_episode_portfolio_values = episode_portfolio_values

            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")


            for key, value in self.q_table.items():
                # Convert each element of the key tuple to a Python int
                clean_key = tuple(int(k) for k in key)
                print(clean_key, value)

        return last_episode_portfolio_values

    def test_agent(self):
        """
        Test the Q-learning agent.
        """
        state, _ = self.test_env.reset(seed=42)
        done = False
        total_reward = 0
        values_over_time = []

        self.exploration_rate = 0  # No exploration during testing

        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            values_over_time.append(info['portfolio_valuation'])

        print(f"Test reward: {total_reward}")
        return total_reward, values_over_time


