"""Offline example: move forward without a model or network access."""


class Agent:
    def configure(self, config):
        self.config = config
        self.settings = config["agent"]

    def reset(self, goal):
        self.goal = goal

    def act(self, observation):
        action = list(observation["noop_action"])
        action[0] = 1
        return action
