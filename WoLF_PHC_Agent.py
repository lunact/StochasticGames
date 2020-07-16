
from Agents import Agent
from Soccer import Soccer
import numpy as np
import random


def roulette(dict):
#def roulette(self, dict):
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


class WoLF_PHC_Agent(Agent):
    """
    A Specific Agent class for WoLF-PHC Agent
    """
    
    def __init__(self,game,playerA,playerB):
        self.g = game
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in game.actions(playerA, state)} \
                     for action_B in game.actions(playerB, state)} \
                 for state in game.states()}
        self.V = {state: 0 for state in game.states()}
        self.pi = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}
        self.pi_bar = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}
        self.C = {state : 0 for state in game.states()}
        #self.k = 0
        self.delta_win = 0.0025
        self.delta_lose = 0.01
        self.delta = self.delta_lose
        #self.s = roulette(game.initial_state())
        self.alpha = 0.9
        self.explor = 0.3


    def play(self,s,playerA,playerB,playerB_policy):
        # 1) voir si on fait greedy ou pas
        a = random.choice(self.g.actions(playerA, s)) if np.random.rand() < self.explor else roulette(self.pi[s])
        # 2) on fait l'action a choisi par 1), on observe la récompense et le nouvel état x
        o = roulette(playerB_policy[s]) #action joueurB
        actions = {playerA: a, playerB: o}
        # on récupère la récompense
        #print("state=",s,"actions =",actions)
        R = self.g.rewards(s, actions)
        rew = R.get(playerA)
        # on observe le nouvel état
        #s2 = roulette(self.g.transition(s, actions))
        return a,o,actions,rew

    def update(self,s,s2,k,a,o,rew,playerA,playerB,playerB_policy):

        self.C[s] += 1

        # 3) on met a jour Q
        self.Q[s][o][a] = (1-self.alpha)*self.Q[s][o][a] + self.alpha*(rew + self.g.gamma()*self.V[s2])

        # 4) on calcule la politique "moyenne"
        for action in self.g.actions(playerA, s):
            self.pi_bar[s][action] = (1/self.C[s])*(self.pi[s][action] - self.pi_bar[s][action])

        # 5) on détermine si on gagne/perd et on choisi delta
        if sum([p*q for p,q in zip(self.pi[s].values(),self.Q[s][o].values())]) > sum([p*q for p,q in zip(self.pi_bar[s].values(),self.Q[s][o].values())]) :
            self.delta = self.delta_win
        else :
            self.delta = self.delta_lose

        # 6) on met à jour la politique pi
        argmax = self.g.actions(playerA, s)[np.argmax(list(self.Q[s][o].values()))]
        for action in self.g.actions(playerA, s):
            if action == argmax:
                self.pi[s][action] = self.pi[s][action] + self.delta
                self.pi[s][action] = min(self.pi[s][action],1.) #pour restreindre à une proba
            else :
                self.pi[s][action] = self.pi[s][action] - self.delta/(len(self.g.actions(playerA, s))-1)
                self.pi[s][action] = max(0.,self.pi[s][action]) #pour restreindre à une proba
    
        # 7) decay et mises à jour
        self.V[s] = self.Q[s][o][max(self.Q[s][o] , key = lambda k : self.Q[s][o][k])]
        self.alpha = 1/(10+(k/10000))
        self.delta_lose = 1/(200000 + k)
        self.delta_win = 4*self.delta_lose
        #self.s = s2
        return self.pi,self.V


#game = Soccer()
#playerA, playerB = game.players()[0], game.players()[1]

#test_agent =  WoLF_PHC_Agent(game, playerA,playerB)
#opponent = WoLF_PHC_Agent(game, playerB,playerA)

#pi,V = test_agent.training(game,playerA,playerB,100,opponent,10)