import numpy as np
import random
from collections import defaultdict
from config import dynamic_features_arr, hyperparams
from models.agent import Agent
from online_normalization import OnlineNormalization
from utils import get_hashasble_state


class QLearningAgent(Agent):
    def __init__(self, train_env, test_env, action_size):
        self.train_env = train_env
        self.test_env = test_env
        self.action_size = action_size
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.exploration_rate = hyperparams['exploration_rate']
        self.exploration_decay = hyperparams['exploration_decay']
        self.exploration_min = hyperparams['exploration_min']

        # Use the ordered min_max_array (array of tuples with min and max for each feature)
        #self.min_max_array = min_max_array
        #self.num_bins = hyperparams['num_bins']  # Use 20 bins as per your request

        # Initialize Q-table (discrete state, action pairs)
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        self.training_reward_last_episode = 0


    def choose_action(self, state):
        """
        Choose an action based on the Q-table or exploration.
        """

        #discrete_state = self._discretize_state(state)
        state = get_hashasble_state(state)
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])


    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning algorithm.
        """
        # discrete_state = self._discretize_state(state)
        # next_discrete_state = self._discretize_state(next_state)
        state = get_hashasble_state(state)
        next_state = get_hashasble_state(next_state)

        current_q = self.q_table[state][action]

        if not done:
            best_future_q = np.max(self.q_table[next_state])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                    reward + self.discount_factor * best_future_q)
        else:
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * reward

        self.q_table[state][action] = new_q

    def train_agent(self):
        """
        Train the Q-learning agent by interacting with the environment.
        """
        last_episode_portfolio_values = []
        last_episode_actions = []  # To store actions taken in the last episode
        n_episodes = hyperparams['n_episodes']

        for episode in range(n_episodes):
            state, _ = self.train_env.reset(seed=42)

            total_reward = 0
            done = False
            episode_portfolio_values = []
            episode_actions = []  # To store actions taken in this episode
            previous_open_price = None  # To store the previous day's open price

            while not done:
                action = self.choose_action(state)  # Choose an action
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated

                current_open_price = info['data_open']  # Get the open price for the current step

                # Update portfolio value using the previous day's open price
                if previous_open_price is not None:
                    portfolio_value = info['portfolio_valuation'] * previous_open_price / info['data_close']
                else:
                    # If this is the first step, just use the initial portfolio value
                    portfolio_value = info['portfolio_valuation']

                # Learn from this step (Q-learning update)
                self.learn(state, action, reward, next_state, done)

                # Update state and tracking variables
                state = next_state
                total_reward += reward
                episode_portfolio_values.append(portfolio_value)  # Track portfolio value for this step
                episode_actions.append(hyperparams['positions'][action])  # Track the action taken

                # Update previous open price for the next step
                previous_open_price = current_open_price

            if episode == n_episodes - 1:
                last_episode_portfolio_values = episode_portfolio_values
                last_episode_actions = episode_actions  # Track actions in the last episode
                self.training_reward_last_episode = total_reward

            # Decay the exploration rate
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

            # Print Q-table for debugging purposes
            for key, value in self.q_table.items():
                clean_key = tuple(int(k) for k in key)
                print(clean_key, value)

        return last_episode_portfolio_values, last_episode_actions

    def test_agent(self):
        pass


    def get_q_values(self, state):
        """
        Return Q-values for all actions based on the given state.
        """
        #discrete_state = self._discretize_state(state)
        state = get_hashasble_state(state)

        return self.q_table[state]


