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

class RL_Agent(Agent):
    """
    Generic class for agents
    """

    def __init__(self,game,playerID):
        self.playerID = playerID
        self.g = game
        self.explor = 0.3
        self.pi = {state: {action: 1/len(game.actions(playerID, state)) for action in game.actions(playerID, state)} for state in game.states()}  


    def compute_policy(self):
        return self.pi

    def play(self,s,playerID):
        #greedy ou non :
        action = random.choice(self.g.actions(playerID, s)) if np.random.rand() < self.explor else roulette(self.pi[s])
        return action 

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):
        """Returns updated policy and V table
        
        :rtype: strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
        raise(NotImplementedError)




