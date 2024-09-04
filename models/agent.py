from abc import ABC, abstractmethod

from config import hyperparams, dynamic_features_arr


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def train_agent(self):
        print(f"Start training with hyperparameters: {hyperparams}")
        print(f"Number of dynamic features: {len(dynamic_features_arr)}")

    @abstractmethod
    def test_agent(self):
        pass
