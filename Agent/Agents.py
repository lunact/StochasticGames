import numpy as np
import random

class Agent:
    """
    Generic class for agents
    """

    def compute_policy(self):
        """ Computes optimal policy for non RL Agents, and returns initial policy for RL Agents
        
        :rtype: strategy (dictionary: {state: {action: probability}})
        """
        raise(NotImplementedError)

    def play(self,s,playerID):
        """ Choice of actions for players
        
        :rtype: player action
        """
        raise(NotImplementedError)

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):
        """Returns updated policy and V table, and (optimal) policy for non RL Agents
        
        :rtype: strategy (dictionary: {state: {action: probability}})
        """
        raise(NotImplementedError)

    def roulette(self,dict):
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

    

