import numpy as np 

class RunningVariance:
    # Keeps a running estimate of variance

    def __init__(self):
        self.m_k = None
        self.s_k = None
        self.k = None

    def add(self, x):
        if not self.m_k:
            self.m_k = x
            self.s_k = 0
            self.k = 0
        else:
            old_mk = self.m_k
            self.k += 1
            self.m_k += (x - self.m_k) / self.k
            self.s_k += (x - old_mk) * (x - self.m_k)

    def get_variance(self, epsilon=1e-12):
        return self.s_k / (self.k - 1 + epsilon) + epsilon
    
    def get_mean(self):
        return self.m_k
    
def get_advantages(values, rewards, gamma=0.999, lmbda=0.95):
    #GAE
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lmbda * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return adv