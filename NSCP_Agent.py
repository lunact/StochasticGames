from RL_Agent import RL_Agent
import numpy as np
import random
from gurobipy import *


class NSCP_Agent(RL_Agent):
    """
    A Specific Agent class for WoLF-PHC Agent
    """
    
    def __init__(self,game,playerID):
        super(NSCP_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.pi2 = {state: {action: 1/len(game.actions(playerID, state)) for action in game.actions((1-self.playerID), state)} for state in game.states()}
        self.n = {state: {action: 0 for action in game.actions((1-self.playerID), state)} for state in game.states()}
        self.C = {state : 0 for state in self.g.states()}
        self.V = {state: 1 for state in self.g.states()}
        self.alpha = 0.9

    def best_response(self,s):
        m = Model()
        m.setParam('OutputFlag', False)
        # 1) convert entrées into array form
        V = m.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="V")
        Q = np.array( [ [self.Q[s][o][a] for o in self.g.actions((1-self.playerID), s)] for a in self.g.actions(self.playerID, s) ] )
        pi2 = np.array([(self.pi2[s][o]) for o in self.g.actions((1-self.playerID), s)])
        # 2) set variables 
        x = np.array([m.addVar(vtype=GRB.CONTINUOUS, name=str(a)) for a in self.g.actions(self.playerID, s)])
        m.update()
        # 3) set objective
        obj = (x.T @ Q @ pi2)
        m.setObjective(obj, GRB.MAXIMIZE)
        # 4) set constraints
        m.addConstr(sum(x) >= 1.0, "")
        m.addConstr(sum(x) <= 1.0, "")
        m.addConstr(sum(x), GRB.EQUAL, 1.0, "")
        for a in x:
            m.addConstr(a >= 0, "")
        # 5) solve
        m.optimize()

        return ({a: x[i].x for i, a in enumerate(self.g.actions(self.playerID, s))} , V.x)

    def learn(self,s,s2,k,a,o,rewA,rewB,opponent_policy):

        self.n[s][o] += 1
        self.C[s] += 1
        
        # 1) on met à jour la politique adversaire estimée
        for action_o in self.g.actions((1-self.playerID), s):
            self.pi2[s][action_o] = self.n[s][action_o]/self.C[s]

        # 2) on met à jour notre politique pi
        self.pi[s],self.V[s] = self.best_response(s)

        # 3) on met a jour Q
        self.Q[s][o][a] = (1-self.alpha)*self.Q[s][o][a] + self.alpha*(rewA + self.g.gamma()*self.V[s2])
    
        return self.pi,self.V