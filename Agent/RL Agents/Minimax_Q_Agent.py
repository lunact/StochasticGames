from RL_Agent import RL_Agent
import numpy as np
import random
from gurobipy import *

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


class Minimax_Q_Agent(RL_Agent):
    """
    A Specific Agent class for Minimax-Q Agent
    """
    
    def __init__(self,game,playerID):
        super(Minimax_Q_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.V = {state: 1 for state in self.g.states()}
        self.decay = .01 ** (1. / ( 2 *10**4))

    def maximin(self, state):
        """
        solve the static game associated with one state of a null sum 2-player stochastic game
    
        :param game: NullSum2PlayerStochasticGame
        :param expected_rewards: dictionary {player B action : {player A action : reward}}
        :param state: state ID
        :rtype: state value (float) and player A strategy (dictionary: {action: probability})
        """
    
        try:
            # Model created
            m = Model("jeu_simple")
            m.setParam('OutputFlag', False) # no console print
            # Variables
            V = m.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="V") # V can be negative
            pi_s = np.array([m.addVar(vtype=GRB.CONTINUOUS, name="p_"+str(a)) for a in self.g.actions(self.playerID, state)])
            m.update()
            # Objective
            m.setObjective(V, GRB.MAXIMIZE)
            # Constraints
            for action1 in self.g.actions((1-self.playerID), state):
                # conversion dictionary > np.array while preserving the same order for actions
                expected_rewards_action1 = np.array([self.Q[state][action1][a] for a in self.g.actions(self.playerID, state)])
                m.addConstr(V <= np.dot(pi_s, expected_rewards_action1), "")
            m.addConstr(sum(pi_s) >= 1.0, "")
            m.addConstr(sum(pi_s) <= 1.0, "")
            m.addConstr(sum(pi_s), GRB.EQUAL, 1.0, "")
            for pi_s_a in pi_s:
                m.addConstr(pi_s_a >= 0, "")
            # Solving
            m.optimize()
            return(V.x, {action0: pi_s[i].x for i, action0 in enumerate(self.g.actions(self.playerID, state))})
    
        except GurobiError:
            print('Error reported')

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):

        self.Q[s][o][a] = (1-self.alpha) * self.Q[s][o][a] + self.alpha * (rewA + self.g.gamma() * self.V[s2])
        self.V[s], self.pi[s] = self.maximin(s)
        
        self.alpha *= self.decay
        return self.pi,self.V
