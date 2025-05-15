class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, **kwargs):
        return self.env.action_space.sample(), None
