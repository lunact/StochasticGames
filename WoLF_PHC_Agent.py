
from RL_Agent import RL_Agent
import numpy as np
import random


class WoLF_PHC_Agent(RL_Agent):
    """
    A Specific Agent class for WoLF-PHC Agent
    """
    
    def __init__(self,game,playerID):
        super(WoLF_PHC_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.V = {state: 0 for state in game.states()}
        self.pi_bar = {state: {action: 1/len(game.actions(playerID, state)) for action in game.actions(self.playerID, state)} for state in game.states()}
        self.C = {state : 0 for state in self.g.states()}
        self.delta_win = 0.0025
        self.delta_lose = 0.01
        self.delta = self.delta_lose
        self.alpha = 0.9
        self.explor = 0.3

    def learn(self,s,s2,k,a,o,rew,opponent_policy):

        self.C[s] += 1

        # 3) on met a jour Q
        self.Q[s][o][a] = (1-self.alpha)*self.Q[s][o][a] + self.alpha*(rew + self.g.gamma()*self.V[s2])

        # 4) on calcule la politique "moyenne"
        for action in self.g.actions(self.playerID, s):
            self.pi_bar[s][action] = (1/self.C[s])*(self.pi[s][action] - self.pi_bar[s][action])

        # 5) on détermine si on gagne/perd et on choisi delta
        if sum([p*q for p,q in zip(self.pi[s].values(),self.Q[s][o].values())]) > sum([p*q for p,q in zip(self.pi_bar[s].values(),self.Q[s][o].values())]) :
            self.delta = self.delta_win
        else :
            self.delta = self.delta_lose

        # 6) on met à jour la politique pi
        argmax = self.g.actions(self.playerID, s)[np.argmax(list(self.Q[s][o].values()))]
        for action in self.g.actions(self.playerID, s):
            if action == argmax:
                self.pi[s][action] = self.pi[s][action] + self.delta
                self.pi[s][action] = min(self.pi[s][action],1.) #pour restreindre à une proba
            else :
                self.pi[s][action] = self.pi[s][action] - self.delta/(len(self.g.actions(self.playerID, s))-1)
                self.pi[s][action] = max(0.,self.pi[s][action]) #pour restreindre à une proba
    
        # 7) decay et mises à jour
        self.V[s] = self.Q[s][o][max(self.Q[s][o] , key = lambda k : self.Q[s][o][k])]
        self.alpha = 1/(10+(k/10000))
        self.delta_lose = 1/(200000 + k)
        self.delta_win = 4*self.delta_lose
        return self.pi,self.V

