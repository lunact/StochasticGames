from non_RL_Agent import non_RL_Agent
import numpy as np
import random

class Random_Agent(non_RL_Agent):
    """
    A Specific Agent class for Random (stationnary uniform policy) Agent
    """

    def __init__(self,game,playerID):
        super(Random_Agent,self).__init__(game,playerID)

    def compute_policy(self):
        return self.pi