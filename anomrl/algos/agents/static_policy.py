class StaticPolicy:
    def __init__(self, env, action):
        self.env = env
        self.action = action

    def predict(self, *args, **kwargs):
        return self.action, None
