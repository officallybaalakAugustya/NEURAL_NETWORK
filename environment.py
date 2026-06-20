class SurvivalEnv:
    def __init__(self):
        pass
    def get_state(self):
        return [0.0, 0.0, 0.0, 0.0]
    def step(self, action):
        return [0.0, 0.0, 0.0, 0.0], 0.0, False
    def render(self):
        pass