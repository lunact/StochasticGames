from RL_Agent import RL_Agent
import numpy as np
import random

def roulette(dict):
    """
    randomly return one of the dictionary's key according to its probability (its value)
    
    :param dict: dictionary {index: probability}
    :rtype: index
    """
    rnd = np.random.rand()
    tmp = 0
    for s, p in dict.items():
        tmp += p
        if tmp >= rnd:
            return s


class Q_Leaning_Agent(RL_Agent):
    """
    A Specific Agent class for Q-Learning Agent
    """
    
    def __init__(self,game,playerID):
        super(Q_Leaning_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.V = {state: 1 for state in self.g.states()}
        self.alpha = 1.
        self.decay = .01 ** (1. / ( 2 *10**4))

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):

        self.Q[s][o][a] = (1-self.alpha) * self.Q[s][o][a] + self.alpha * (rewA + self.g.gamma() * self.V[s2])
        self.V[s], self.pi[s] = max(self.g, self.Q[s], s)
        
        self.alpha *= self.decay
        return self.pi,self.V