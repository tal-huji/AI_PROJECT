# models/simple_agent.py

class SimpleAgent:
    def __init__(self, dynamic_features_arr, action_mapping):
        """
        Initialize the Simple Agent.
        :param dynamic_features_arr: List of dynamic feature functions to be applied to the history.
        :param action_mapping: Dictionary mapping dynamic feature outputs to actions.
        """
        self.dynamic_features_arr = dynamic_features_arr
        self.action_mapping = action_mapping

    def get_action(self, history):
        """
        Determine the action based on the dynamic features' output.
        :param history: Historical price data.
        :return: The action corresponding to the dynamic features output.
        """
        # Compute dynamic features output
        dynamic_feature_output = tuple(int(feature(history)) for feature in self.dynamic_features_arr)

        # Select action based on the dynamic features output
        action = self.action_mapping.get(dynamic_feature_output, 0)  # Default to neutral (0) if no match
        return action

    def train_agent(self):
        """
        This agent doesn't train like the Q-Learning or DQN agents.
        """
        return [], []

    def test_agent(self, env):
        """
        Test the agent on the environment by following the simple strategy.
        :param env: Test environment
        :return: total_reward, portfolio_values_over_time, test_actions
        """
        state, _ = env.reset(seed=42)
        done = False
        total_reward = 0
        portfolio_values_over_time = []
        actions_over_time = []

        while not done:
            # Compute the action based on the environment history
            history = env.get_history()  # Assuming env provides historical data
            action = self.get_action(history)
            actions_over_time.append(action)

            # Step through the environment using the selected action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            portfolio_values_over_time.append(info['portfolio_valuation'])

        print(f"Simple Agent Test Reward: {total_reward}")
        return total_reward, portfolio_values_over_time, actions_over_time


# Action mapping for dynamic features output
action_mapping = {
    (0, 0, 0): 0,    # Neutral
    (1, 0, 0): 1,    # Small rise detected in short-term, buy small
    (0, 1, 0): 1,    # Small rise in medium-term, buy small
    (0, 0, 1): 1,    # Small rise in long-term, buy small
    (1, 1, 0): 2,    # Rise in both short and medium, buy
    (0, 1, 1): 2,    # Rise in both medium and long, buy
    (1, 0, 1): 2,    # Rise in both short and long, buy
    (2, 0, 0): 2,    # Strong rise in short-term, buy aggressively
    (0, 2, 0): 2,    # Strong rise in medium-term, buy aggressively
    (0, 0, 2): 2,    # Strong rise in long-term, buy aggressively
    (1, 1, 1): 2,    # Moderate rise across all terms, buy aggressively
    (2, 2, 2): 2,    # Strong rise across all terms, buy aggressively
    (1, 2, 1): 1,    # Mixed rise, buy moderately
    (2, 1, 0): 2,    # Strong rise in short, buy aggressively
    (0, 1, 2): 2,    # Strong rise in medium and long, buy aggressively
    (1, 2, 2): 2,    # Strong rise in medium and long, buy aggressively
    (2, 1, 1): 2,    # Strong rise in short, buy aggressively
    (2, 2, 1): 2,    # Strong rise in short and medium, buy aggressively
    (0, 2, 2): 2,    # Strong rise in medium and long, buy aggressively
    (2, 0, 2): 2,    # Strong rise in short and long, buy aggressively
}

