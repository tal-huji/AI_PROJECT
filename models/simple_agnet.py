import numpy as np
from itertools import product
from collections import defaultdict

from mpmath import hyper

from config import hyperparams
from models.agent import Agent
from utils import get_hashasble_state


class SimpleAgent(Agent):
    def __init__(self, train_env, test_env, dynamic_features_arr):
        """
        Initialize the Simple Agent.
        :param train_env: Training environment.
        :param test_env: Test environment.
        :param dynamic_features_arr: List of dynamic feature functions to be applied to the state.
        :param bin_mapping: Dictionary mapping dynamic features to their corresponding bins.
        """
        self.train_env = train_env
        self.test_env = test_env
        self.dynamic_features_arr = dynamic_features_arr
        self.training_reward_last_episode = 0
        self.positions = hyperparams['positions']

    def default_action_logic(self, state):
        """
        Default logic for mapping dynamic feature output to actions based on hyperparams['positions'].
        :param state: A tuple of dynamic feature outputs.
        :return: An action from the available positions.
        """
        total = sum(state)

        # Select the appropriate action based on total feature values and available positions
        num_positions = len(self.positions)

        # For 2-position systems (e.g., [0, 1])
        if num_positions == 2:
            return self.positions[0] if total <= 1 else self.positions[1]

        # For 3-position systems (e.g., [-1, 0, 1])
        elif num_positions == 3:
            if total <= 1:
                return self.positions[1]  # Neutral or hold
            elif total == 2:
                return self.positions[2]  # Buy
            else:
                return self.positions[0]  # Short position

        # Default to neutral if no valid positions found
        return self.positions[1]

    def choose_action(self, state):
        """
        Choose an action based on the dynamic features' output.
        :param state: Current state of the environment, already in dynamic feature terms.
        :return: The action corresponding to the dynamic features output.
        """
        state = get_hashasble_state(state)
        return self.default_action_logic(state)


    def train_agent(self):
        """
        This agent doesn't train like the Q-Learning agent. It uses a rule-based approach.
        :return: Empty lists for portfolio values and actions (since it doesn't learn over time).
        """
        last_episode_portfolio_values = []
        last_episode_actions = []

        # Since this agent doesn't learn, there's no episodic training to track
        print("Simple Agent does not require training.")

        return last_episode_portfolio_values, last_episode_actions

    def test_agent(self):
        """
        Test the agent on the environment by following the simple strategy.
        :return: total_reward, portfolio_values_over_time, test_actions
        """
        state, _ = self.test_env.reset(seed=42)
        done = False
        total_reward = 0
        portfolio_values_over_time = []
        actions_over_time = []

        while not done:
            # Choose an action based on the current state
            action = self.choose_action(state)
            actions_over_time.append(action)

            # Step through the environment using the selected action
            next_state, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated

            # Update state and portfolio tracking
            state = next_state
            total_reward += reward
            portfolio_values_over_time.append(info['portfolio_valuation'])

        print(f"Simple Agent Test Reward: {total_reward}")
        return total_reward, portfolio_values_over_time, actions_over_time

    def learn(self, state, action, reward, next_state, done):
        """
        Since this agent doesn't learn, this method is a no-op.
        """
        pass

    def get_q_values(self, state):
        """
        Since this agent doesn't use a Q-table, return None.
        """
        return None
