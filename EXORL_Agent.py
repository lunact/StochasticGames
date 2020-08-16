from RL_Agent import RL_Agent
import numpy as np
import random
from gurobipy import *


class EXORL_Agent(RL_Agent):
    """
    A Specific Agent class for WoLF-PHC Agent
    """
    
    def __init__(self,game,playerID):
        super(EXORL_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.Q2 = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions((1-self.playerID), state)} \
                     for action_B in self.g.actions(self.playerID, state)} \
                 for state in self.g.states()}
        self.V = {state: 0 for state in self.g.states()}
        self.V2 = {state: 0 for state in self.g.states()}
        self.pi2 = {state: {action: 1/len(game.actions(playerID, state)) for action in game.actions((1-self.playerID), state)} for state in game.states()}
        self.alpha = 0.9
        self.beta = 0.6
        self.tuning_param = 0.2

    def extended_optimal_response(self,s):
        m = Model()
        m.setParam('OutputFlag', False)
        m.setParam('NonConvex', 2)
        # 1) convert entrées into matrix form
        Q = np.array( [ [self.Q[s][o][a] for o in self.g.actions((1-self.playerID), s)] for a in self.g.actions(self.playerID, s) ] )
        Q2 = np.array( [ [self.Q2[s][a][o] for a in self.g.actions(self.playerID, s)] for o in self.g.actions((1-self.playerID), s) ] )
        pi2 = np.array([(self.pi2[s][o]) for o in self.g.actions((1-self.playerID), s)])
        # 2) set variables
        x = np.array([m.addVar(vtype=GRB.CONTINUOUS, name=str(a)) for a in self.g.actions(self.playerID, s)])
        y = np.array([m.addVar(vtype=GRB.CONTINUOUS, name=str(o)) for o in self.g.actions((1-self.playerID), s)])
        m.update()
        # 3) set objective
        obj = (x.T @ Q @ pi2) - self.tuning_param*((y.T @ Q2 @ x) - (pi2.T @ Q2 @ x))
        m.setObjective(obj, GRB.MAXIMIZE)
        # 4) set constraints
        m.addConstr(sum(x) >= 1.0, "")
        m.addConstr(sum(x) <= 1.0, "")
        m.addConstr(sum(x), GRB.EQUAL, 1.0, "")
        m.addConstr(sum(y) >= 1.0, "")
        m.addConstr(sum(y) <= 1.0, "")
        m.addConstr(sum(y), GRB.EQUAL, 1.0, "")

        for a in x:
            m.addConstr(a >= 0, "")
        for o in y:
            m.addConstr(o >= 0, "")

        # 5) solve
        m.optimize()
        return {a: x[i].x for i, a in enumerate(self.g.actions(self.playerID, s))}

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):

        # 1) on met a jour Q
        self.Q[s][o][a] = (1-self.alpha)*self.Q[s][o][a] + self.alpha*(rewA + self.g.gamma()*self.V[s2])
        self.Q2[s][a][o] = (1-self.alpha)*self.Q2[s][a][o] + self.alpha*(rewB + self.g.gamma()*self.V2[s2])

        # 2) on met à jour la politique adversaire estimée
        for action_o in self.g.actions((1-self.playerID), s) :
            self.pi2[s][action_o] = (1-self.beta)*self.pi2[s][action_o]
            self.pi2[s][o] += self.beta

        # 3) on met à jour notre politique pi
        self.pi[s] = self.extended_optimal_response(s)
    
        # 4) mise à jour V et V2
        self.V[s] = self.Q[s][o][max(self.Q[s][o] , key = lambda k : self.Q[s][o][k])]
        self.V2[s] = self.Q2[s][a][max(self.Q2[s][a] , key = lambda k : self.Q2[s][a][k])]
        return self.pi,self.V
