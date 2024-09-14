from config import hyperparams


class PriceComparisonAgent:
    def __init__(self):
        """
        Initialize the Price Comparison Agent.
        """
        self.positions = hyperparams['positions']  # Assume [sell, hold]
        self.action = None  # This will store the chosen action after training (either hold or sell)
        self.initial_price = None
        self.final_price = None

    def train_agent(self, env_train):
        """
        Train the agent by comparing the initial and final prices of the training period.
        This will determine the action (either hold or sell) for the test period.
        :param env_train: The environment used for training (contains price data).
        """
        # Reset the environment and track the first and last prices
        state, _ = env_train.reset()
        done = False
        step_count = 0

        while not done:
            next_state, reward, terminated, truncated, info = env_train.step(0)  # Dummy step to progress the environment
            done = terminated or truncated

            if step_count == 0:
                self.initial_price = info['data_close']  # Record the initial price at the start of training

            step_count += 1

        self.final_price = info['data_close']  # Record the final price at the end of the training

        # Decide action based on price comparison
        if self.final_price > self.initial_price:
            self.action = self.positions[1]  # Hold
        else:
            self.action = self.positions[0]  # Sell

        print(f"Training completed. Initial price: {self.initial_price}, Final price: {self.final_price}, Action: {self.action}")

    def test_agent(self, env_test):
        """
        No-op for testing. The action is already determined during training.
        :param env_test: The environment used for testing (not used here).
        """
        pass

    def choose_action(self, state):
        """
        Choose an action based on the price comparison.
        :param state: Current state of the environment (not used here).
        :return: The action determined during training.
        """
        return self.action
