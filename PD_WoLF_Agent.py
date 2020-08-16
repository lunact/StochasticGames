from RL_Agent import RL_Agent
import numpy as np
import random


class PD_WoLF_Agent(RL_Agent):
    """
    A Specific Agent class for PD-WoLF Agent
    """
    
    def __init__(self,game,playerID):
        super(PD_WoLF_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.V = {state: 0 for state in self.g.states()}
        self.Delta = {state: {action: 0 for action in game.actions(playerID, state)} for state in game.states()}
        self.Delta2 = {state: {action: 0 for action in game.actions(playerID, state)} for state in game.states()}
        self.delta_win = 0.0025
        self.delta_lose = 0.01
        self.delta = self.delta_lose
        self.alpha = 0.9

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):
        
        # 1) on met a jour Q
        self.Q[s][o][a] = (1-self.alpha)*self.Q[s][o][a] + self.alpha*(rewA + self.g.gamma()*self.V[s2])

        # 2) on détermine si on gagne/perd et on choisi delta
        if self.Delta[s][a]*self.Delta2[s][a] < 0:
            self.delta = self.delta_win
        else :
            self.delta = self.delta_lose

        # 3) on met à jour la politique pi (et les differences)
        argmax = self.g.actions(self.playerID, s)[np.argmax(list(self.Q[s][o].values()))]
        for action in self.g.actions(self.playerID, s):
            if action == argmax:
                self.pi[s][action] = self.pi[s][action] + self.delta
                self.pi[s][action] = min(self.pi[s][action],1.) #pour restreindre à une proba
                self.Delta2[s][action] = self.delta - self.Delta[s][action]
                self.Delta[s][action] = self.delta
            else :
                D = self.delta/(len(self.g.actions(self.playerID, s))-1)
                self.pi[s][action] = self.pi[s][action] - D
                self.pi[s][action] = max(0.,self.pi[s][action]) #pour restreindre à une proba
                self.Delta2[s][action] = D - self.Delta[s][action] #might be -D ? :p
                self.Delta[s][action] = D #might be -D ? :p
    
        # 4) decay et mises à jour
        self.V[s] = self.Q[s][o][max(self.Q[s][o] , key = lambda k : self.Q[s][o][k])]
        self.alpha = 1/(10+(k/10000))
        self.delta_lose = 1/(200000 + k)
        self.delta_win = 4*self.delta_lose
        return self.pi,self.V

