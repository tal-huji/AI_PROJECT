from collections import defaultdict
import numpy as np
import random
from config import dynamic_features_arr, hyperparams
from models.agent import Agent
from online_normalization import OnlineNormalization


class QLearningAgent(Agent):
    def __init__(self, train_env, test_env, action_size):
        self.train_env = train_env
        self.test_env = test_env
        self.n_bins = hyperparams['num_bins']
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_decay = hyperparams['exploration_decay']
        self.exploration_min = hyperparams['exploration_min']

        # Fixed bins for discretization
        self.bins = [np.linspace(-1, 1, self.n_bins + 1) for _ in range(len(dynamic_features_arr))]

        # Q-table initialized using defaultdict
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        # Initialize online normalization
        self.normalizer = OnlineNormalization(len(dynamic_features_arr))

    def _get_discrete_state(self, state):
        """
        Convert continuous state to a discrete state using pre-defined fixed bins.
        Handle potential multidimensional state arrays.
        """
        if isinstance(state, np.ndarray):
            state = state.flatten()  # Flatten the array if it's multidimensional
        elif isinstance(state, list):
            state = np.array(state).flatten()  # Convert to numpy array and flatten

        # Update the normalization statistics and normalize the state
        self.normalizer.update(state)
        normalized_state = self.normalizer.normalize(state)

        discretized = self.discretize_state(normalized_state, self.bins)
        return tuple(discretized)  # Return as tuple for indexing into the Q-table

    def discretize_state(self, state, bins):
        """
        Discretize continuous feature values into discrete bins.
        """
        return tuple(np.digitize(feature, bin_edges) - 1 for feature, bin_edges in zip(state, bins))

    def choose_action(self, state):
        """
        Choose an action based on the Q-table or exploration.
        """
        discrete_state = self._get_discrete_state(state)
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            return int(np.argmax(self.q_table[discrete_state]))

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning algorithm.
        """
        discrete_state = self._get_discrete_state(state)
        next_discrete_state = self._get_discrete_state(next_state)

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
