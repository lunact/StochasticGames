from Agents import Agent
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

class non_RL_Agent(Agent):
    """
    Generic class for agents
    """

    def __init__(self,game,playerID):
        self.playerID = playerID
        self.g = game
        self.pi = {state: {action: 1/len(game.actions(self.playerID, state)) for action in game.actions(self.playerID, state)} for state in game.states()}  

    def compute_policy(self):
        """ Computes optimal policy
        
        :rtype: strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
        raise(NotImplementedError)

    def play(self,s,playerID):
        action = self.roulette(self.pi[s])
        return action

    def learn(self,s,s2,k,a,o,rew,opponent_policy): #old update
        return self.pi


