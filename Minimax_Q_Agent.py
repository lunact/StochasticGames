from Agents import Agent
import StaticGameResolution as sgr
from WoLF_PHC_Agent import WoLF_PHC_Agent
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


class Minimax_Q_Agent(Agent):
    """
    A Specific Agent class for Minimax-Q Agent
    """
    
    def __init__(self,game,playerA,playerB):
        self.g = game
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in game.actions(playerA, state)} \
                     for action_B in game.actions(playerB, state)} \
                 for state in game.states()}
        self.V = {state: 1 for state in game.states()}
        self.pi = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}
        #self.k = 0
        #self.s = roulette(game.initial_state())
        self.alpha = 1.
        self.explor = 0.3
        self.decay = .01 ** (1. / ( 2 *10**4))

    def play(self,s,playerA,playerB,playerB_policy):
        # choose an action
        a = random.choice(self.g.actions(playerA, s)) if np.random.rand() < self.explor else roulette(self.pi[s])
        o = roulette(playerB_policy[s])
        actions = {playerA: a, playerB: o}
        # learn
        R = self.g.rewards(s, actions)
        rew = R.get(playerA)
        return a,o,actions,rew

    def update(self,s,s2,k,a,o,rew,playerA,playerB,playerB_policy):
        #self.k += 1
        #s2 = roulette(self.g.transition(self.s, actions))

        self.Q[s][o][a] = (1-self.alpha) * self.Q[s][o][a] + self.alpha * (rew + self.g.gamma() * self.V[s2])
        self.V[s], self.pi[s] = sgr.maximin(self.g, self.Q[s], s, playerA, playerB)
        
        #self.s = s2
        self.alpha *= self.decay
        return self.pi,self.V


#game = Soccer()
#playerA, playerB = game.players()[0], game.players()[1]
#test_agent =  Minimax_Q_Agent(game, playerA,playerB)

#opponent = WoLF_PHC_Agent(game, playerB,playerA)

#pi,V = test_agent.training(game,playerA,playerB,100,opponent,10)