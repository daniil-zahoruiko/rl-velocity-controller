import numpy as np

class OU:
    def __init__(self, scale, mean, variance):
        self.scale = scale
        self.mean = mean
        self.variance = variance
        self.reset()

    def reset(self):
        self.x = np.copy(self.mean)

    def sample(self):
        self.x += self.scale * (self.mean - self.x) + np.random.normal(loc=0.0, scale=np.sqrt(self.variance), size = self.mean.shape)
        return self.x